from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
import sys, os
current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_folder)
import k_converse, k_train

def get_profile_from_text(text):
    lines = text.split('\n')
    profile = []
    prefix = 'your persona:'
    for line in lines:
        temp = line.strip()
        if temp.startswith(prefix):
            text = temp[len(prefix): ].strip()
            profile.append(text)
    return profile, lines[-1].strip()


def is_training_state(obs):
    if 'label' in obs:
        return True
    return False



class KMethod(Agent):
    @staticmethod
    def add_param():
        return
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt = self.opt
        self.history = {}
        self.batch_idx = shared and shared.get('batchindex') or 0

        self.profile = [] # profile of user
        self.chat_history = [] # log chat from the past
        self.next_new_episode = True # if the next is profile-text

        if shared:
            self.opt = shared['opt']
            self.metrics = shared['metrics']
        else:
            self.metrics = {'loss': 0.0}
        self.reset()

    def update_profile(self, observation):
        text = observation['text']
        episode_done = observation['episode_done']
        if self.next_new_episode:
            self.profile.clear()
            self.chat_history.clear()
            p_texts, question = get_profile_from_text(text)
            if len(p_texts) == 0:
                print ('EXCEPTION: start of episode profile is empty for text: %s' % text)
            self.profile = []
            for item in p_texts:
                temp_item = k_train.preprocess_rm_period(item)
                self.profile.append(temp_item.split(' ')) # self.dict.txt2vec(item)
            self.chat_history = []
            self.next_new_episode = False
            observation['text'] = question
        if episode_done:  # if this is the last text in this episode, set a flag for the next
            self.next_new_episode = True
        if 'labels' in observation:
            label_text = observation['labels']
            self.chat_history.append(label_text) # self.dict.txt2vec(label_text)
        elif 'eval_labels' in observation:
            self.chat_history.append((observation['text'], observation['eval_labels'][0]))
        observation['profile'] = list(self.profile)
        observation['history'] = list(self.chat_history)


    def share(self):
        shared = super().share()
        shared['opt'] = self.opt
        shared['metrics'] = self.metrics
        return shared

    def observe(self, observation):
        obs = observation.copy()
        if 'text' not in obs:
            obs['text'] = None
            #print ('non obs: ', obs)
        else:
            self.update_profile(obs)
            question = obs['text']
        self.observation = obs
        return obs

    def act(self):
        return self.batch_act([self.observation])[0]

    def batch_act(self, observations):
        """
        text_candidates: to eval hit
        text: to eval perplexity and F1
        :param observations:
        :return:
        """
        #print (json.dumps(observations, ensure_ascii=False))
        #print ('=====================================')

        batchsize = len(observations)
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]
        for i in range(batchsize):
            obs = observations[i]
            if obs['text'] is None:
                continue
            profile = obs['profile']
            obs = observations[i]
            if 'label_candidates' in obs:
                cands = obs['label_candidates']
                converse = {'question': obs['text'], 'cand': cands, 'profile': profile}
                indices = k_converse.get_ranked_indices_for_one_question(converse)
                ranked_res = [cands[index] for index in indices]
                batch_reply[i]['text_candidates'] = ranked_res
            batch_reply[i]['text'] = 'hello' #label_cands[0]
            if 'eval_labels' in obs:
                label_text = obs['eval_labels'][0]
                #label_text_toks = self.dict.txt2vec(label_text)
                #self.model.train(profile, history, text_toks, label_text_toks, [])
        #print ('observe: ', observations[0])
        #print ('act: ', batch_reply[0])
        return batch_reply

    def save(self, path=None):
        return


    def shutdown(self):
        return


    def load(self, path):
        return

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        self.history.clear()
        self.reset_metrics()

    def reset_metrics(self):
        """Reset metrics for reporting loss and perplexity."""
        self.metrics['loss'] = 0.0