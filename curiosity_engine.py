from dotenv import load_dotenv
from llama_utils import *
import numpy as np
import os, sys

load_dotenv()
URL = os.environ['URL']


class Curiosity:
	def __init__(self, llama_host: str):
		self.api = setup_client(llama_host)
		self.map = {'observe': 'gemma3:4b',
					'reflect': 'deepseek-r1:8b',
					'imagine': 'gemma3:4b',
					'curiosity': 'gemma3:4b',
					'wisdom': 'gemma3:4b'}
		self.focus = []
		self.inner_dialogue = []
		self.refs = []

	def observe(self, subject):
		# What is interesting about subject? What can be discovered?
		self.focus.append(subject)
		seek_observations = (f"I will provide a topic, and I want you to find what is interesting about it and what could"
							f" potentially be discovered. The topic is: {subject}")
		raw = ask_model(self.api, self.map['observe'], seek_observations)
		self.inner_dialogue.append(raw.message.content)
		return raw.message.content.split('</think>')[-1]

	def reflect(self, category_landscape):
		ref = []
		# refine how self.focus can be broken into possible avenues of thought from category_landscape
		segments = category_landscape.split('---')
		postulates = segments[0].split(':')
		for c in postulates:
			# look for ones with many tokens
			n_tokens = len(c)
			most_verbose = 0
			for idea in c.split('?'):
				ratio = len(idea) / n_tokens * 100
				if ratio > 20:
					ref.append(idea + '?')
					most_verbose = ratio
				elif len(count_starred(idea)) > 2:
					ref.append(idea)
		self.refs = ref
		return '[REFLECTIONS]:'+f'\n-'.join(ref)

	def simulate(self, reflection):
		# imagine and explore paths given
		imagined_results = ''
		imodel = self.map['imagine']
		pre_prompt = (f'You will be given a series of ideas. I want to you analyze the overall pattern and validity of the'
		              f'ideas. Then for each idea imagine ways to either answer or deepen the understanding of our focus'
		              f': *{self.focus[0]}*. For each idea you explore preface with [IMAGINED]. Heres the ideas:'
		              f'\n{reflection}')
		ideas = ask_model(self.api, imodel, pre_prompt)
		theories = []
		for concept in ideas.message.content.split('[IMAGINED]'):
			score = len(concept.splitlines())
			for ln in concept.splitlines():
				n_starred = count_starred(ln)
				if score > 4 and len(n_starred) > 4:
					theories.append(concept)
					continue
		prompting = (f'Attempting to understand "{self.focus} some interesting topics arise. I will provide a list of '
		             f'ideas and your job is to distill them into questions that will elevate the understanding or lead to new'
		             f'ideas. Please start each question with [QUESTION]. Here are the ideas we have to work with:\n')
		return prompting + '\n-'.join(theories)

	def generate_question(self, possible_ideas):
		# given the results of rumination pick the most interesting ideas
		model = self.map['curiosity']
		interests = ask_model(self.api,model,possible_ideas).message.content
		return interests.split('</think>')[1:]

	def curious_alchemy(self, questions):
		#Assess what the true nature of that idea is. Also consider why it might not be accurate or useful,etc.
		prompting = (f'Attempting to understand "{self.focus} some interesting topics arise. I will provide a list of '
		             f'ideas and your job is to distill them into questions that will elevate the understanding or lead to new'
		             f'ideas. Please start each question with [QUESTION]. Here are the ideas we have to work with:\n{questions}')
		insight = ask_model(self.api, self.map['curiosity'],prompting).message.content
		return insight.split('[QUESTION]')[1:]



def count_starred(phrase):
	starred_words = []
	for word in phrase.split(' '):
		star_count = np.array([letter == '*' for letter in word], dtype=np.int8).sum()
		if star_count >= 2:
			starred_words.append(word)
	return starred_words


def main():
	conscious = True
	# Define a subject. Ideally a complex and rich question that is precisely worded, but conceptually is nearly unbounded
	subject = 'What makes something real?'
	if len(sys.argv)>1:
		subject = ' '.join(sys.argv[1:])
	c = Curiosity(URL)
	i = 0
	limit = 7
	h = []
	try:
		while conscious and i < limit:
			print(f'[~] Observing...')
			signal = c.observe(subject)
			reflection = c.reflect(signal)
			imagined_paths = c.simulate(reflection)
			next_question = c.generate_question(imagined_paths)
			print(f'[~] New Questions:\n{subject}')
			subject = c.curious_alchemy(next_question)
			print(f'='*180)
			i += 1
			n = ' '
			banner = f'\n[~] New Questions(s): {f"{n}".join(subject)}\n' + '='*80
			h.append(f'{banner}{f"{n}".join(subject)}')
	except KeyboardInterrupt:
		pass


if __name__ == '__main__':
	main()
