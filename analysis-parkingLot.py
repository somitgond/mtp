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
packetSize = 1400 * 8 # packet size in bytes (approx)

def avg_throughput_calc(folder_path, debug=0):
    filename = folder_path + "flowmonitor.xml"
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

    avg_throughput = np.mean(throughput_data)
    std_throughput = np.std(throughput_data)
    min_throughput = max(avg_throughput - std_throughput, 0)
    max_throughput = min(avg_throughput + std_throughput, 2)

    avg_goodput = np.mean(goodput_data)
    std_goodput = np.std(goodput_data)
    min_goodput = max(avg_goodput - std_goodput, 0)
    max_goodput = min(avg_goodput+ std_goodput, 2)

    avg_fct = np.mean(fct_data)
    std_fct = np.std(fct_data)
    min_fct = max(avg_fct - std_fct, 0)
    max_fct = avg_fct + std_fct

    total_data_sent = np.sum(transmitted_data)
    
    return (avg_throughput,
            std_throughput,
            min_throughput,
            max_throughput,
            avg_goodput,
            std_goodput,
            min_goodput,
            max_goodput,
            avg_fct,
            std_fct,
            min_fct,
            max_fct,
            total_data_sent)

def packet_loss(folder_path, debug=0):
    tree = ET.parse(f"{folder_path}/flowmonitor.xml")
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

    avg_pkt_loss = np.mean(pkt_loss) * 100
    std_pkt_loss = np.std(pkt_loss) * 100
    min_pkt_loss = max(avg_pkt_loss - std_pkt_loss, 0)
    max_pkt_loss = min(avg_pkt_loss + std_pkt_loss, 100)

    return (avg_pkt_loss,
            std_pkt_loss,
            min_pkt_loss,
            max_pkt_loss)

# different for both bottlenecks
def compute_link_utilization( folder_path, delta_time=0.1):
    file_path1 = folder_path + "bottleneckTx-parkingLot-1.txt"
    file_path2 = folder_path + "bottleneckTx-parkingLot-2.txt"
    return (compute_link_utilization_helper(file_path1), compute_link_utilization_helper(file_path2))

def compute_link_utilization_helper(file_path):
    times, packets = [], []

    # Read cumulative packets with timestamps
    with open(file_path, "r") as f:
        for line in f:
            t, pkt = line.strip().split()
            times.append(float(t))
            packets.append(int(pkt))
    times = np.array(times)
    packets = np.array(packets)
    #
    # last_pkt = packets[np.size(packets)-1]
    # resultant_size = np.size(packets)
    # for i in range(np.size(packets)-1, 0, -1):
    #     if packets[i] != last_pkt:
    #         resultant_size = i+1
    #         break
    #
    # packets = np.resize(packets, resultant_size)
    # times = np.resize(times, resultant_size)
    #
    # # remove trailing zeroes
    # packet = np.trim_zeros(packets, 'b')
    # times = np.resize(times, np.size(packet))
    # print(f"Queue: {times[np.size(times)-1]}")
    #
    # Differences
    delta_t = np.diff(times)                 # interval durations
    delta_packets = np.diff(packets)         # packets sent in each interval

    # Convert to throughput (bps)
    bits_transmitted = delta_packets * 8
    throughput_bps = bits_transmitted / delta_t

    # Utilization = throughput / link_capacity
    utilization_percent = (throughput_bps / bottleneckLinkBandwidth) * 100

    # Mean and std
    mean_util = np.mean(utilization_percent)
    std_util = np.std(utilization_percent)
    min_util = max(mean_util - std_util, 0)
    max_util = min(mean_util + std_util, 100)

    return (mean_util, std_util, min_util, max_util)

# different for both bottleneck links
# finding effective delay
def effective_delay(folder_path, debug=0):
    filename_rtt = folder_path + "RTTs.txt"
    filename_qsize1 = folder_path + "tc-qsizeTrace-parkingLot-1.txt"
    filename_qsize2 = folder_path + "tc-qsizeTrace-parkingLot-2.txt"

    # reading data
    rtt_data = np.genfromtxt(filename_rtt, delimiter=" ").reshape(-1, 2)
    queue_data1 = np.genfromtxt(filename_qsize1, delimiter=" ").reshape(-1, 2)
    queue_data2 = np.genfromtxt(filename_qsize2, delimiter=" ").reshape(-1, 2)

    # queue_data1_col = queue_data1[:, 1]
    # queue_data2_col = queue_data2[:, 1]
    # last_nonzero_idx_q1 = np.flatnonzero(queue_data1_col)[-1]
    # last_nonzero_idx_q2 = np.flatnonzero(queue_data2_col)[-1]
    # new_q_size = max(last_nonzero_idx_q2, last_nonzero_idx_q1)
    #
    # queue_data1 = queue_data1[:new_q_size+1, :]
    # queue_data2 = queue_data2[:new_q_size+1, :]
    # print(f"t1: {queue_data1[new_q_size]}")
    # print(f"t2: {queue_data2[new_q_size]}")

    # if dubugging is on print data values
    if debug == 1:
        print(rtt_data)
        print(queue_data1)
        print(queue_data2)

    return (effective_delay_helper(rtt_data, queue_data1), effective_delay_helper(rtt_data, queue_data2))


def effective_delay_helper(rtt_data, queue_data):
    # for 1 is added in queue buffer
    queue_data[:, 1] += 1  #FIXME: read from file 

    tempQdata = (((queue_data[:, 1]) * packetSize * 8) / bottleneckLinkBandwidth)
    # average effective delay and jitter
    combined = 2 + np.mean(rtt_data[:, 1]) + tempQdata
    jitter_avg_rtt = np.var(combined)
    avg_rtt = np.mean(combined)

    # queuing delay
    queuing_delay = np.mean(tempQdata)
    std_queuing_delay = np.std(tempQdata)
    min_queuing_delay = max(queuing_delay - std_queuing_delay, 0)
    max_queuing_delay = queuing_delay + std_queuing_delay

    return (avg_rtt, jitter_avg_rtt, queuing_delay, std_queuing_delay, min_queuing_delay, max_queuing_delay)
    

if __name__ == "__main__":
    # added source ip address
    for i in range(0, 60):
        SOURCE_IPS.append(f"10.1.{i}.1")

    random_seeds = [ 69713, 56629, 86799, 42653, 82842, 72958, 23256, 14590,
                    98472, 8288, 42653, 42653, 42653, 42653, 42653, 42653,
                    42653, 42653, 42653, 42653, ]

    rtts = list(np.arange(150,305,5))
    src_path = "results-parkingLot_d4/"
    file_name_to_file_index = [
        ["results_parkingLot_LN_aqm_zc_150_to_300_mlBeta", 0],
        # ["results_parkingLot_LN_codel_zc_150_to_300_mlBeta", 31],
        ["results_parkingLot_LN_naqm_150_to_300_mlBeta", 31],
        ["results_parkingLot_LN_red_150_to_300_mlBeta", 62],
    ]

    fields = [
        "SimNumber", "randomSeed",
        "RTT",
        "avgThroughput", "stdThroughput", "minThroughput", "maxThroughput",
        "avgGoodput", "stdGoodput", "minGoodput", "maxGoodput",
        "flowCompTime", "stdFlowCompTime", "minFlowCompTime", "maxFlowCompTime",
        "avgDataSent",
        "linkUtilization1", "stdLinkUtilization1", "minLinkUtilization1", "maxLinkUtilization1",
        "linkUtilization2", "stdLinkUtilization2", "minLinkUtilization2", "maxLinkUtilization2",
        "effectiveDelay", "jitterRTT",
        "queuingDelay1", "stdQueuingDelay1", "minQueuingDelay1", "maxQueuingDelay1",
        "queuingDelay2", "stdQueuingDelay2", "minQueuingDelay2", "maxQueuingDelay2",
        "packetLoss", "stdPktLoss", "minPktLoss", "maxPktLoss", ]
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

            file_name = f"{src_path}/result-parkingLot-{i}"

            cmd_to_run = f"tar -zxf {file_name}.gzip -C {src_path}"

            # run the command
            subprocess.run(cmd_to_run, shell=True, stdout=subprocess.DEVNULL)
            print(f"\tExtracted {file_name}.gzip")
            time.sleep(2)

            folder_path = file_name + "/"

            ((eff_rtt1, jitter1, queue_delay1, std_q_delay1, min_q_delay1, max_q_delay1),
             (eff_rtt2, jitter2, queue_delay2, std_q_delay2, min_q_delay2, max_q_delay2)) = effective_delay(folder_path)
            avg_throughput, std_throughput, \
            min_throughput, max_throughput, avg_goodput, std_goodput,  \
            min_goodput, max_goodput, avg_fct, std_fct, min_fct, max_fct,  \
            avg_data_sent = avg_throughput_calc(folder_path)

            pkt_loss, std_pkt_loss, min_pkt_loss, max_pkt_loss = packet_loss(folder_path)
            ((avg_lu1, std_lu1, min_lu1, max_lu1),
             (avg_lu2, std_lu2, min_lu2, max_lu2)) = compute_link_utilization(folder_path)

            data_to_write = [
                num,
                #random_seeds[i - start_file_index],
                42653,
                rtts[i - start_file_index],
                avg_throughput, std_throughput, min_throughput, max_throughput,
                avg_goodput, std_goodput, min_goodput, max_goodput,
                avg_fct, std_fct, min_fct, max_fct,
                avg_data_sent,
                avg_lu1, std_lu1, min_lu1, max_lu1,
                avg_lu2, std_lu2, min_lu2, max_lu2,
                eff_rtt1, jitter1,
                queue_delay1, std_q_delay1, min_q_delay1, max_q_delay1,
                queue_delay2, std_q_delay2, min_q_delay2, max_q_delay2,
                pkt_loss, std_pkt_loss, min_pkt_loss, max_pkt_loss,
            ]

            # write data in output file
            with open(data_filename, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data_to_write)
            num += 1
