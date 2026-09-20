# HV-AECE-Based-Monolithic-Self-Powered-Cardiac-Monitoring-System
Design and measurement artifacts for an AECE-based self-powered cardiac monitoring system.
The relevent CSV files for my measurement results is put in the release of this github respository.
# RR-CNN Functional Reference
This repository provides the training, functional reference, and verification files for a lightweight R-R interval CNN classifier.

- `train_rr_cnn_binary_lite_lite.py`: CNN training script
- `rr_classifier_verify_with_flatten.py`: Python functional reference and test-vector generator
- `rr_cnn_functional_reference.sv`: SystemVerilog functional CNN reference
- `testbench_flatten.sv`: SystemVerilog verification testbench

The network uses 20 R-R intervals as input and performs binary classification through two 1-D convolution layers followed by pooling and a fully connected layer.

The provided SystemVerilog code is a functional reference model for layer-level and numerical verification, and does not disclose the implementation-specific microarchitecture of the on-chip CNN accelerator.
