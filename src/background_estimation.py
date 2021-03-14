import cv2
import numpy as np

class AdaptativeBackgroundModel():
	def __init__(self, height, width):
		self.mean = np.zeros((height, width))
		self.std = np.zeros((height, width))

	def estimate_background(self, training_frames):
		for k, frame in enumerate(training_frames):
			delta = frame - self.mean
			self.mean += delta / (k+1)
			delta_std = frame - self.mean
			self.std += delta * delta_std
		self.std = np.sqrt(self.std/len(training_frames))
		return self.mean, self.std

	def compute_fg_and_bg(self, frame, alpha):
		fg = np.abs(frame-self.mean) >= (alpha * (self.std +2))
		bg = ~fg
		return fg, bg

	def adapt_background(self, frame, bg, rho):
		self.mean[bg] = rho * frame[bg] + (1 - rho) * self.mean[bg]
		self.std[bg] = np.sqrt(
			rho * np.power(frame[bg] - self.mean[bg], 2) + (1-rho) * np.power(self.std[bg], 2)
		)
		return self.mean
	
	def evaluate(self, eval_frames, alpha, rho):
		foreground_frames = []
		background_frames = []
		for k, frame in enumerate(eval_frames):
			fg, bg = self.compute_fg_and_bg(frame, alpha)
			new_bg = self.adapt_background(frame, bg, rho)
			foreground_frames.append((fg*255).astype(np.uint8))
			background_frames.append((new_bg).astype(np.uint8))
		return foreground_frames, background_frames




