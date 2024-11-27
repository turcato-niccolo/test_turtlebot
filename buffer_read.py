import pickle
import numpy as np

with open('replay_buffer.pkl', 'rb') as f:
    replay_buffer = pickle.load(f)

    if np.isnan(replay_buffer.state.any()):
        print(replay_buffer.state)
    
    if np.isnan(replay_buffer.next_state.any()):
        print(replay_buffer.next_state)
    
    if np.isnan(replay_buffer.action.any()):
        print(replay_buffer.action)
    
    if np.isnan(replay_buffer.reward.any()):
        print(replay_buffer.reward)

    if np.isnan(replay_buffer.not_done.any()):
        print(replay_buffer.not_done)