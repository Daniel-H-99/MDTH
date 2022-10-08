from pipelines.th import THPipeline

class THTest:

	def __init__(self, gpus, twin_id=None, test_video=None, test_image=None, output_name=None, preprocess_drv=1):
		self.test_gpus = list(map(lambda x: int(x), gpus.split(',')))
		self.test_twin_id = 't_01824827-cb8f-7c76-8020-67f0366636dd'
		self.test_video = '/mnt/warping-shared/warping-data/th_model/video/30_00000.mp4'
		self.test_audio = '/mnt/warping-shared/warping-data/th_model/audio/00000.wav'
		self.test_image = '/mnt/warping-shared/warping-data/th_model/img/00000.png'
		self.config = 'config.yml'
		self.output_name = output_name
		self.preprocess_drv = preprocess_drv
		if twin_id is not None:
			self.test_twin_id = twin_id
		if test_video is not None: 
			self.test_video = test_video
		if test_image is not None:
			self.test_image = test_image

		self.test()
		



	def test(self):
		
		for gpu_id in self.test_gpus:
			th = THPipeline(config=self.config, gpus=self.test_gpus)
			th.set_twin(twin_id=self.test_twin_id) 	

			if self.preprocess_drv:
				# test preprocess	
				print(f'base video_path = {self.test_video}')
				th.initialize(task='preprocess')
				th.preprocess(base_video_path=self.test_video)

			# test inference
			th.initialize(task='inference')
			th.inference(input_audio_path=self.test_audio, 
					        input_image_path=self.test_image,
							output_name=self.output_name,
					        is_high_resolution=1)
			break
