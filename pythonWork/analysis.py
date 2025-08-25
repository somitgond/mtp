# author: Somit Gond
# date: 02/04/2025

"""
Calculate metric on given data 
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

# some default values
bottleneckLinkBandwidth = 100 * 1000 * 1000 # in bits / second
packetSize = 1454 * 8 # packet size in bytes (approx)

def avg_throughput_calc(folder_path, debug=0):
    filename = folder_path + "dumbbell-flowmonitor.xml"
    # throughput calculation
    tree = ET.parse(filename)
    root = tree.getroot()
    flowstats = root[0]

    # flow data (metadata)
    attri = []
    for flows in flowstats:
        attri.append(flows.attrib)

    # one-to-one mapping between client and destination
    flows_ip = {}
    for ips in root[1]:
        temp = ips.attrib
        flows_ip[temp["flowId"]] = [
            temp["sourceAddress"],
            temp["destinationAddress"],
        ]

    return calculate_throughput(attri, flows_ip, debug)

def calculate_throughput(flow_data, flows_ip, debug=0):
    throughput_data = []
    fct_data = []
    transmitted_data = []
    goodput_data = []

    for flow in flow_data:
        flow_id = flow["flowId"]

        # traffic originate from source ip
        if flows_ip[flow_id][0] not in SOURCE_IPS:
            continue

        rx_bytes = int(flow["rxBytes"])  # Total received bytes on the flow end 
        time_first_tx_ns = float(
            flow["timeFirstTxPacket"].replace("+", "").replace("ns", "")
        )  # First transmission time (ns)
        time_last_rx_ns = float(
            flow["timeLastRxPacket"].replace("+", "").replace("ns", "")
        )  # Last Recieved time (ns)

        # Calculate total time in seconds
        total_time_sec_fct = (time_last_rx_ns - time_first_tx_ns) / 1e9

        # Calculate throughput in Mbps
        if(total_time_sec_fct != 0):
            throughput_bps = rx_bytes / total_time_sec_fct
            # approximate 40 bytes of header
            goodput_bps = (rx_bytes - (40 * int(flow["rxPackets"]))) / total_time_sec_fct
        else:
            throughput_bps = 0

        # value of 1Mb = 1e6
        throughput_mbps = (throughput_bps * 8) / 1e6 
        goodput_mbps = (goodput_bps * 8) / 1e6

        data_sent = (rx_bytes * 8) / 1e6

        if debug == 1:
            print(f"Flow: {flow_id} throughput: {throughput_mbps}mbps")
            print(f"Flow: {flow_id} Goodput: {goodput_mbps}mbps")

        throughput_data.append(throughput_mbps)
        goodput_data.append(goodput_mbps)
        fct_data.append(total_time_sec_fct)
        transmitted_data.append(data_sent)

    throughput_data = np.array(throughput_data)
    fct_data = np.array(fct_data)
    transmitted_data = np.array(transmitted_data)
    return (
        np.mean(throughput_data),
        np.std(throughput_data),
        np.mean(goodput_data),
        np.std(goodput_data),
        np.mean(fct_data),
        np.std(fct_data),
        np.mean(transmitted_data),
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

def compute_link_utilization(
    folder_path, delta_time=0.1, packet_size_bytes=1454, link_bandwidth_mbps=100.0
):
    file_path = folder_path + "bottleneckTx-dumbbell.txt"
    packets = []

    # Read the cumulative packet log
    with open(file_path, "r") as f:
        for line in f:
            t, pkt = line.strip().split()
            packets.append(int(pkt))

    packets = np.array(packets)

    # Get packets per second
    delta_packets = np.diff(packets)
    delta_packets = delta_packets[delta_packets != 0]

    # Packets per second
    pps = delta_packets / delta_time

    throughput_mbps = pps * packet_size_bytes * 8 / 1e6 # 1Mb = 1e6

    # Utilization = throughput / link bandwidth
    utilization_percent = (throughput_mbps / link_bandwidth_mbps) * 100

    # Mean and std
    mean_util = np.mean(utilization_percent)
    std_util = np.std(utilization_percent)

    return mean_util, std_util

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

    tempQdata = (((queue_data[:, 1]) * packetSize * 8) / bottleneckLinkBandwidth)
    # average effective delay and jitter
    combined = 2 + np.mean(rtt_data[:, 1]) + tempQdata
    jitter_avg_rtt = np.var(combined)
    avg_rtt = np.mean(combined)

    # queueing delay
    queueing_delay = np.mean(tempQdata)
    std_queuing_delay = np.std(tempQdata)

    return avg_rtt, jitter_avg_rtt, queueing_delay, std_queuing_delay


if __name__ == "__main__":
    # added source ip address
    for i in range(0, 60):
        SOURCE_IPS.append(f"10.1.{i}.1")

    fields = [
        "Simulation_number",
        "Random Seed",
        "RTT",
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
    rtts = list(np.arange(150,305,5))
    src_path = "results-withRed"
    file_name_to_file_index = [
        ["results_LN_aqm_zc_150_to_300", 0],
        ["results_LN_codel_zc_150_to_300", 31],
        ["results_LN_naqm_150_to_300", 62],
        ["results_LN_red_150_to_300", 93],
        #["results_LN_aqm_zc_TM", 40],
        #["results_LN_codel_zc_TM", 60],
        #["results_LN_aqm_zc_100MB", 80],
        #["results_LN_codel_zc_100MB", 100],
        #["results_LN_aqm_zc_100MB_TM", 120],
        #["results_LN_codel_zc_100MB_TM", 140],
    ]

    for fn, start_file_index in file_name_to_file_index:
        print(f"Running for: {fn}")
        data_filename = f"{fn}.csv"
        with open(data_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(fields)

        num = 0
        step_size = 31;
        # change step size based on simulation
        for i in range(start_file_index, start_file_index + step_size):
            print(f"\tIteration: {num}, rtt={rtts[i-start_file_index]}")

            file_name = f"{src_path}/result-clientServerRouter-{i}"

            cmd_to_run = f"tar -zxf {file_name}.gzip -C {src_path}"

            # run the command
            subprocess.run(cmd_to_run, shell=True, stdout=subprocess.DEVNULL)
            print(f"\tExtracted {file_name}.gzip")
            time.sleep(2)

            folder_path = file_name + "/"

            eff_rtt, jitter, queue_delay, std_queue_delay = effective_delay(folder_path)
            throughput_avg, std_throughput, goodput_avg, goodput_std, fct_avg, std_fct, data_avg = (
                avg_throughput_calc(folder_path)
            )

            pkt_loss, std_pkt_loss = packet_loss(folder_path)
            lu_avg, lu_std = compute_link_utilization(folder_path)

            data_to_write = [
                num,
                #random_seeds[i - start_file_index],
                42653,
                rtts[i - start_file_index],
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

            # write data in output file 
            with open(data_filename, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data_to_write)
            num += 1
