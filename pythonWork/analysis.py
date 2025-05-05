# author: Somit Gond
# date: 02/04/2025

"""
This file is used to simulate ns3 script and find global sync threshold
Flow:
run simulation T times
each time calculate avg throughput and average global synchronization
"""

import csv
import os
import random
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from re import sub

import numpy as np

SOURCE_IPS = []


def avg_throughput_calc(folder_path, fct=False, debug=0):
    # os.chdir(folder_path)
    filename = folder_path + "dumbbell-flowmonitor.xml"
    # throughput calculation
    tree = ET.parse(filename)
    root = tree.getroot()
    flowstats = root[0]
    attri = []
    for flows in flowstats:
        attri.append(flows.attrib)

    flows_ip = {}
    for ips in root[1]:
        temp = ips.attrib
        flows_ip[temp["flowId"]] = [
            temp["sourceAddress"],
            temp["destinationAddress"],
        ]

    return calculate_throughput(attri, flows_ip, fct, debug)


def calculate_throughput(flow_data, flows_ip, fct=False, debug=0):
    throughput_data = []
    fct_data = []
    trasmitted_data = []

    for flow in flow_data:
        flow_id = flow["flowId"]

        if flows_ip[flow_id][0] not in SOURCE_IPS:
            continue

        tx_bytes = int(flow["txBytes"])  # Transmitted bytes
        time_first_tx_ns = float(
            flow["timeFirstTxPacket"].replace("+", "").replace("ns", "")
        )  # First transmission time (ns)
        time_last_tx_ns = float(
            flow["timeLastTxPacket"].replace("+", "").replace("ns", "")
        )  # Last transmission time (ns)

        time_last_rx_ns = float(
            flow["timeLastRxPacket"].replace("+", "").replace("ns", "")
        )  # Last Recieved time (ns)

        # Calculate total time in seconds
        total_time_sec = (time_last_tx_ns - time_first_tx_ns) / 1e9

        total_time_sec_fct = (time_last_rx_ns - time_first_tx_ns) / 1e9

        # Calculate throughput in Mbps
        throughput_bps = tx_bytes / total_time_sec
        throughput_mbps = (throughput_bps * 8) / (1024 * 1024)

        data_sent = (tx_bytes * 8) / (1024 * 1024)

        if debug == 1:
            print(f"Flow: {flow_id} throughput: {throughput_mbps}mbps")

        throughput_data.append(throughput_mbps)
        fct_data.append(total_time_sec_fct)
        trasmitted_data.append(data_sent)

    throughput_data = np.array(throughput_data)
    fct_data = np.array(fct_data)
    trasmitted_data = np.array(trasmitted_data)
    return (
        np.mean(throughput_data),
        np.std(throughput_data),
        np.mean(fct_data),
        np.std(fct_data),
        np.mean(trasmitted_data),
    )


def packet_loss(folder_path, debug=0):
    tree = ET.parse(f"{folder_path}dumbbell-flowmonitor.xml")
    root = tree.getroot()
    flowstates = root[0]
    flow_data = []
    for flows in flowstates:
        flow_data.append(flows.attrib)

    # map flow id to source ip address
    flow_id_to_src_ip = {}
    for i in root[1]:
        temp = i.attrib
        flow_id_to_src_ip[temp["flowId"]] = temp["sourceAddress"]

    # calculate packet loss percentage
    lost_pkts = []
    pkts_sent = []
    for flow in flow_data:
        if flow_id_to_src_ip[flow["flowId"]] not in SOURCE_IPS:
            continue
        lost_pkts.append(int(flow["lostPackets"]))
        pkts_sent.append(int(flow["txPackets"]))
        if debug == 1:
            print(f"Packet Lost: {lost_pkts[-1]} total packets: {pkts_sent[-1]}")

    lost_pkts = np.array(lost_pkts)
    pkts_sent = np.array(pkts_sent)

    pkt_loss = lost_pkts / pkts_sent

    return np.mean(pkt_loss) * 100, np.std(pkt_loss) * 100


def mean_goodput(folder_path, debug=0):
    tree = ET.parse(f"{folder_path}/dumbbell-flowmonitor.xml")
    root = tree.getroot()
    flowstates = root[0]  # FlowStats
    classifier = root[1]  # FlowClassifiers

    # Parse flow data
    flow_data = [flow.attrib for flow in flowstates]

    # Map flowId to source IP
    flow_id_to_src_ip = {}
    for classifier_entry in classifier:
        attrs = classifier_entry.attrib
        flow_id_to_src_ip[attrs["flowId"]] = attrs["sourceAddress"]

    # Dictionary to store total bits and durations per sender IP
    sender_bits = defaultdict(float)
    sender_times = defaultdict(float)

    for flow in flow_data:
        flow_id = flow["flowId"]

        if flow_id_to_src_ip[flow_id] not in SOURCE_IPS:
            continue

        src_ip = flow_id_to_src_ip[flow_id]
        rx_bytes = int(flow["rxBytes"])

        time_first_tx_ns = float(
            flow["timeFirstTxPacket"].replace("+", "").replace("ns", "")
        )  # First transmission time (ns)
        time_last_tx_ns = float(
            flow["timeLastTxPacket"].replace("+", "").replace("ns", "")
        )  # Last transmission time (ns)

        # Calculate total time in seconds
        duration = (time_last_tx_ns - time_first_tx_ns) / 1e9
        if duration > 0:
            sender_bits[src_ip] += rx_bytes * 8.0  # bits
            sender_times[src_ip] += duration
            if debug:
                print(
                    f"Flow {flow_id} from {src_ip}: duration = {duration:.3f}s, rxBytes = {rx_bytes}"
                )

    # Compute goodput in Mbps for each sender
    goodputs_mbps = []
    for sender in sender_bits:
        goodput = sender_bits[sender] / sender_times[sender]  # bits/sec
        goodputs_mbps.append(goodput / (1024 * 1024))  # Convert to Mbps
        if debug:
            print(f"Sender {sender} - Goodput: {goodput / 1e6:.2f} Mbps")

    # Mean and standard deviation
    if goodputs_mbps:
        mean_gp = np.mean(goodputs_mbps)
        std_gp = np.std(goodputs_mbps)
    else:
        mean_gp = std_gp = 0.0

    return mean_gp, std_gp


def compute_link_utilization(
    folder_path, packet_size_bytes=1454, link_bandwidth_mbps=100.0
):
    file_path = folder_path + "bottleneckTx-dumbbell.txt"
    times = []
    packets = []

    # Read the cumulative packet log
    with open(file_path, "r") as f:
        for line in f:
            t, pkt = line.strip().split()
            times.append(float(t))
            packets.append(int(pkt))

    times = np.array(times)
    packets = np.array(packets)

    # Get packets per second
    delta_time = np.diff(times)
    delta_packets = np.diff(packets)

    # Packets per second
    pps = delta_packets / delta_time

    # Convert to Mbps: (pps × packet_size_bytes × 8) / 1024*1024
    throughput_mbps = pps * packet_size_bytes * 8 / (1024 * 1024)

    # Utilization = throughput / link bandwidth
    utilization_percent = (throughput_mbps / link_bandwidth_mbps) * 100

    # Mean and std
    mean_util = np.mean(utilization_percent)
    std_util = np.std(utilization_percent)

    return mean_util, std_util


def global_sync_value(folder_path, debug=0):
    # define parameters
    window_size = 25  # 5 second window

    # synchrony calculation
    data = []
    for f in os.listdir(folder_path):
        if f.endswith(".cwnd"):
            if debug == 1:
                print(f"Reading file: {folder_path+f}")
            d = np.genfromtxt(folder_path + f, delimiter=" ").reshape(-1, 2)
            data.append(d[:, 1])

    ## convert array to numpy
    data = np.array(data)

    data_loss = np.zeros(data.shape)

    ## convert it into loss events
    for i in range(len(data)):
        for j in range(1, len(data[i])):
            if data[i][j - 1] > data[i][j]:
                data_loss[i][j] = 1

    ## global sync
    sync_rate = []
    for k in range(len(data_loss[0])):
        nij = 0
        low = max(0, k - window_size)
        high = min(k + window_size + 1, len(data_loss[0]))
        for i in range(len(data_loss)):
            for j in range(low, high):
                if data_loss[i][j] == 1:
                    nij += 1
                    break
        sync_rate.append(nij / len(data_loss))
    if debug == 1:
        print(sync_rate)
    sync_rate = np.array(sync_rate)
    return np.mean(sync_rate)


# finding effective delay
def effective_delay(folder_path, debug=0):
    filename_rtt = folder_path + "RTTs.txt"
    filename_qsize = folder_path + "tc-qsizeTrace-dumbbell.txt"

    # reading data
    rtt_data = np.genfromtxt(filename_rtt, delimiter=" ").reshape(-1, 2)
    queue_data = np.genfromtxt(filename_qsize, delimiter=" ").reshape(-1, 2)

    # if dubugging is on print data values
    if debug == 1:
        print(rtt_data)
        print(queue_data)

    # for 1 is added in queue buffer
    queue_data[:, 1] += 1

    # average effective delay
    avg_rtt = np.mean(rtt_data[:, 1])
    avg_rtt += (np.mean(queue_data[:, 1]) * 8) / 10**5
    avg_rtt += 2

    # find jitter
    combined = 2 + np.mean(rtt_data[:, 1]) + (queue_data[:, 1] * 8) / 10**5
    jitter_avg_rtt = np.var(combined)

    # queueing delay
    queueing_delay = (np.mean(queue_data[:, 1]) * 8) / 10**5
    std_queuing_delay = (np.std(queue_data[:, 1]) * 8) / 10**5

    return avg_rtt, jitter_avg_rtt, queueing_delay, std_queuing_delay


if __name__ == "__main__":
    # added source ip address
    for i in range(0, 60):
        SOURCE_IPS.append(f"10.1.{i}.1")

    fields = [
        "Simulation_number",
        "Random Seed",
        "RTT",
        "Global Sync Value",
        "Average Throughput(Mbps)",
        "std avg throughput",
        "Average Goodput(Mbps)",
        "std goodput",
        "Link Utilization",
        "std link utilization",
        "Flow Completion Time(s)",
        "std flow comp time(s)",
        "Averate Data Sent(Mb)",
        "Effective Delay(ms)",
        "Jitter in RTT(ms)",
        "Queuing Delay(ms)",
        "std queuing delay",
        "Packet loss %",
        "std pkt loss",
    ]

    random_seeds = [
        69713,
        56629,
        86799,
        42653,
        82842,
        72958,
        23256,
        14590,
        98472,
        8288,
        42653,
        42653,
        42653,
        42653,
        42653,
        42653,
        42653,
        42653,
        42653,
        42653,
    ]
    rtts = [
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        205,
        210,
        215,
        220,
        225,
        230,
        235,
        240,
        245,
    ]
    src_path = "results-3"
    file_name_to_file_index = [
        ["a_results_LN_aqm_zc", 70],
        ["a_results_LN_naqm_zc", 90],
        ["a_results_LN_aqm_zc_100MB", 150],
        ["a_results_LN_naqm_zc_100MB", 170],
        ["a_results_NR_aqm_zc", 190],
        ["a_results_NR_naqm_zc", 210],
        ["a_results_NR_aqm_zc_100MB", 230],
        ["a_results_NR_naqm_zc_100MB", 250],
    ]

    for fn, start_file_index in file_name_to_file_index:
        print(f"Running for: {fn}")
        data_filename = f"{fn}.csv"
        with open(data_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(fields)

        num = 0
        for i in range(start_file_index, start_file_index + 20):
            print(f"\tIteration: {num}")

            file_name = f"{src_path}/result-clientServerRouter-{i}"

            cmd_to_run = f"tar -zxf {file_name}.gzip -C {src_path}"

            # run the command
            subprocess.run(cmd_to_run, shell=True, stdout=subprocess.DEVNULL)
            print(f"\tExtracted {file_name}.gzip")
            time.sleep(2)

            folder_path = file_name + "/"

            # write data in output file
            eff_rtt, jitter, queue_delay, std_queue_delay = effective_delay(folder_path)
            throughput_avg, std_throughput, fct_avg, std_fct, data_avg = (
                avg_throughput_calc(folder_path)
            )
            pkt_loss, std_pkt_loss = packet_loss(folder_path)

            goodput_avg, goodput_std = mean_goodput(folder_path)
            lu_avg, lu_std = compute_link_utilization(folder_path)

            data_to_write = [
                num,
                random_seeds[i - start_file_index],
                rtts[i - start_file_index],
                global_sync_value(folder_path),
                throughput_avg,
                std_throughput,
                goodput_avg,
                goodput_std,
                lu_avg,
                lu_std,
                fct_avg,
                std_fct,
                data_avg,
                eff_rtt,
                jitter,
                queue_delay,
                std_queue_delay,
                pkt_loss,
                std_pkt_loss,
            ]

            with open(data_filename, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data_to_write)
            num += 1
