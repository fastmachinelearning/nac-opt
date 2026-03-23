# Data

There are two versions available:

- [Quantum Machine](https://urldefense.proofpoint.com/v2/url?u=https-3A__purdue0-2Dmy.sharepoint.com_-3Af-3A_g_personal_oyesilyu-5Fpurdue-5Fedu_EuhbLM-2DwFApNiX9Mh5ZMeIEBG3dGqSIPgwN21j5S30nxvQ-3Fe-3DCDc3Xi&d=DwMFAg&c=gRgGjJ3BkIsb5y6s49QqsA&r=3tXuppM5Ux2UBnxU0DCrdSagIS9IpvGOlIFtsYfyWuc&m=5R-PzD5Udxkr2BBA9AYXREVhYselyKDYk_-1g6QMka_dPV3VTCVJe4id5PFOgpLq&s=fUu9yFLybrPN_AYcDhfBiQoXf5RlOAwbo6DmsD3CiqU&e=)
- QICK ZCU216
  - [Version 20230529](https://purdue0-my.sharepoint.com/personal/oyesilyu_purdue_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Foyesilyu%5Fpurdue%5Fedu%2FDocuments%2FQubit%20Readout%20%2D%20Purdue%20%2D%20New%20Data&ga=1)
    - [Partitioned](https://www.dropbox.com/scl/fo/i30pf90fpingvc2o87yrf/h?rlkey=8wfkli0nin11bnnc5ynf457g1&dl=0)
  - [Version 20240501](#)

## Single Qubit Data

### QICK (ZCU216)

Description: 
Readout time is 2000 ns with 2.6 ns sampling rate.

- Train split [0.9]
  - ```X shape```: (909000, 1440)
  - ```y shape```: (909000, 2)
- Test split [0.1]
  - ```X shape```: (101000, 1440)
  - ```y shape```: (101000, 2)

<img src="../images/qick-data.png" alt="drawing" width="500"/>

### Quantum Machine

Contains two subsets: single and 2-qubit mulitplex data.

Description :
Real raw data. Readout time is 2000 ns with 1 sample taken every nanosecond.

- File : **00002_IQ_plot_raw.h5**
- Train split [0.9]
  - ```X shape``` : (9000, 2000)
  - ```y shape``` : (9000, 2)
- Test split [0.1]
  - ```X shape``` : (1000, 2000)
  - ```y shape``` : (1000, 2)
