#!/bin/bash
kubectl apply -f yaml/run_true_monarch_fast.yaml
kubectl apply -f yaml/run_true_monarch.yaml
kubectl apply -f yaml/run_vsa_baseline.yaml