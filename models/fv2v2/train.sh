python run_trainer.py --config config/mesh_fc_stage3_v3_u.yaml --device_ids 0,1 --stage 3 --checkpoint '/mnt/hdd/minyeong_workspace/checkpoints/MDTH/mesh_fc_stage3_u/c00000179-checkpoint.pth.tar'

cd test
bash cmd_s2u_hetero.sh