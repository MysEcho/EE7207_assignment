from trainer import RBFPipeline

pipeline = RBFPipeline(mode="inference", num_centers=20, max_epochs=100, learning_rate=0.01, ckpt_name='best_checkpoint_centers_20')
pipeline.run()