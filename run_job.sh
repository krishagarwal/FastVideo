#!/bin/bash
kubectl apply -f yaml/run_monarch.yaml
kubectl apply -f yaml/run_monarch_dynamic.yaml
kubectl apply -f yaml/run_vsa_baseline.yaml
