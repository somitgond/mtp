# results.tar.gz
it contains the simulation results, every simulation is ran 20 times. First 10 times with random seeds, next 10 times with varying RTT.


|start index | end index | aqm (0 = no, 1 = yes) | data (0 = unbonded) (MB)| TCP variant (0 =  linux reno, 1 = new reno) | 
|-|-|-|-| - | 
| 70 | 89 | 1 | 0| 0 |
|90|109 | 0 | 0 | 0 | 
| 110 | 129 | 1 | 45 | 0 |
| 130 | 149 | 0 | 45 | 0  |
| 150 | 169 | 1 | 100 | 0 |
| 170 | 189 | 0 | 100 | 0 |
| 190 | 209 | 1 | 0 | 1 |
| 210 | 229 | 0 | 0 | 1 |
| 230 | 249 | 1 | 100 | 1 | 
| 250 | 269 | 0 | 100 |  1 | 


# Analysis

post experiment analysis is done using analysis.py
graph plotting is in aqm_V_naqm.ipynb
