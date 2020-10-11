# SIAM
Scalable In-Memory Acceleration With Mesh: Device, Circuits, Architecture, and Algorithm

A comprehensive tool that allows for performance estimation of In-Memory computing (IMC) architectures. The tool is a customized version of NeuroSim (developed by Georgia Tech team led by Prof. Shimeng Yu), combined with a interconnect simulator and a post-mapping accuracy estimator. This tool incorporates device, circuit, architecture, and algorithm properties of in-memory computing. It provides both post-mapping accuracy and hardware performance of IMC architectures.
The current version has software and hardware tools separate and will be combined in the coming versions. This version supports Tensorflow (<2.0 version). 
Software features: Supports quantization, variation-aware training, and hardware-aware training. It incorporates crossbar size, ADC precision, device variations (RRAM only), and algorithm quantization (PACT quantizer).
Hardware Features: Incorporates device, circuit, and architecture evaluation and network-on-chip (NoC) evaluation. The current version supports NoC-mesh and will be extended to other topologies.

If you are using this tool, please cite the following work:
[1] Krishnan, Gokul, Sumit K. Mandal, Chaitali Chakrabarti, Jae-sun Seo, Umit Y. Ogras, and Yu Cao. "Interconnect-aware area and energy optimization for in-memory acceleration of DNNs." IEEE Design & Test (2020).
[2] Mandal, Sumit K., Gokul Krishnan, Chaitali Chakrabarti, Jae-Sun Seo, Yu Cao, and Umit Y. Ogras. "A Latency-Optimized Reconfigurable NoC for In-Memory Acceleration of DNNs." IEEE Journal on Emerging and Selected Topics in Circuits and Systems 10, no. 3 (2020): 362-375.

References:
[1] Chen, Pai-Yu, Xiaochen Peng, and Shimeng Yu. "NeuroSim: A circuit-level macro model for benchmarking neuro-inspired architectures in online learning." IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 37, no. 12 (2018): 3067-3080.
[2] Choi, Jungwook, Zhuo Wang, Swagath Venkataramani, Pierce I-Jen Chuang, Vijayalakshmi Srinivasan, and Kailash Gopalakrishnan. "Pact: Parameterized clipping activation for quantized neural networks." arXiv preprint arXiv:1805.06085 (2018).

Acknowledgement: 
This work was supported in part by C-BRIC, one of the six centers in JUMP, a Semiconductor Research Corporation program sponsored by DARPA, NSF CAREER Award CNS-1651624 and Semiconductor Research Corporation under task ID 2938.001.
