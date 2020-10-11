# SIAM
Scalable In-Memory Acceleration With Mesh: Device, Circuits, Architecture, and Algorithm

A comprehensive tool that allows for performance estimation of In-Memory computing (IMC) architectures. The tool is built on top of NeuroSim (developed by Georgia Tech team led by Prof. Shimeng Yu). This tool incorporates device, circuit, architecture, and algorithm properties of in-memory computing. It provides both post-mapping accuracy and hardware peroformance of IMC architectures.
Current Version has softwre and hardware tools separate and will be combined in comign versions. This version supports Tensorflow (<2.0 version). 
Algorithm features: Supports quantization, variation-aware training, and hardware aware trainging. It incorporates crossbar size, ADC precision, device  variations (RRAM only), and algorithm quanitzation (PACT quantizer).
Hardware Features: Incorporates device, circuit, and architecture evaluation and network-on-chip (NoC) evaluation. Current version supports NoC-mesh and will be extended to other topologies.

If you are usin gthis tool please cite the following work:
[1] Krishnan, Gokul, Sumit K. Mandal, Chaitali Chakrabarti, Jae-sun Seo, Umit Y. Ogras, and Yu Cao. "Interconnect-aware area and energy optimization for in-memory acceleration of DNNs." IEEE Design & Test (2020).

References:
[1] Chen, Pai-Yu, Xiaochen Peng, and Shimeng Yu. "NeuroSim: A circuit-level macro model for benchmarking neuro-inspired architectures in online learning." IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 37, no. 12 (2018): 3067-3080.
[2] Choi, Jungwook, Zhuo Wang, Swagath Venkataramani, Pierce I-Jen Chuang, Vijayalakshmi Srinivasan, and Kailash Gopalakrishnan. "Pact: Parameterized clipping activation for quantized neural networks." arXiv preprint arXiv:1805.06085 (2018).
