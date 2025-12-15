import os
import yaml
import torch

class Config:
	def __init__(self, config_path=None):
		if config_path is None:
			config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
		with open(config_path, 'r') as f:
			cfg = yaml.safe_load(f)
		self.paths = type('Paths', (), cfg['paths'])
		self.feature_extraction = type('FeatureExtraction', (), cfg['feature_extraction'])
		self.model = type('Model', (), cfg['model'])
		self.training = type('Training', (), cfg['training'])
		self.inference = type('Inference', (), cfg['inference'])
		self.evaluation = type('Evaluation', (), cfg['evaluation'])
		self.device = cfg.get('device', 'cuda')

	def get_device(self):
		if self.device == 'cuda' and torch.cuda.is_available():
			return 'cuda'
		return 'cpu'

def get_config():
	return Config()

