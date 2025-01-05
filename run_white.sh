python3 -m RobFR.benchmark.FGSM_white --distance=l2 --goal=dodging --model=MobileFace --eps=16 --dataset=lfw --device=cpu --batch_size=1 --steps=3
#python3 -m RobFR.benchmark.BIM_white --distance=l2 --goal=dodging --model=MobileFace --eps=16 --dataset=lfw --device=cpu --batch_size=1 --iters=10 --steps=3
#python3 -m RobFR.benchmark.MIM_white --distance=l2 --goal=dodging --model=MobileFace --eps=16 --dataset=lfw --device=cpu --batch_size=1 --iters=10 --steps=3 --mu=0.5
#python3 -m RobFR.benchmark.CW_white --goal=dodging --model=MobileFace --eps=16 --dataset=lfw --device=cpu --batch_size=1 --iters=10 --steps=3 --bin_steps=5 --confidence=0.01
