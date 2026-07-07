or Nonlinear local ID plot in the report, can you include another version that does not show linear PCA and d_model, just wnat to see how much TwoNN and MLE agree with each other.

To make my work relatable to other people's work. I want to focus on the AE study that other people did. I have dumped a colleage's work at /mars-vol/marsv/dir3_manifold/autoencoder_share.tar.gz
This work should be done in a dedicated workspace. you can use util files in this dir folder for sure. You should report this study as REPORT_AE.md and only plugged in high level results in REPORT.md
First maybe try if there is any factor that can make AE reconstruction error show an elbow in FVU and/or reconstruction error like his work
Right now the major difference is 
1. switch to Qwen3-1.7B
2. Qwen3-1.7B layer 2 and/or layer2
3. last token only
4. Training cache uses HuggingFaceFW/fineweb-edu, sample-10BT; CE eval uses HuggingFaceFW/fineweb, sample-10BT
5. seq_len=10
6. Much larged AE, 
   encoder:
   2048 → 4096 → 4096 → 2048 → k

   bottleneck:
   k

   decoder:
   k → 2048 → 4096 → 4096 → 2048
7. k-sweep
8. Layer-2 sweep: [5,10,15,20,25,30]; layer-10 sweep: often [5,10,20,40] or fixed k=32 scaling

If only one factor does not work, try to find a controlled experiment that show when it shows an elbow and when it does not.
