

class LightSignalFSM:
    """
    A finite state machine to manage the states of a light signal.
    """

    ALLOWED_NEXT_STATES = {
        'RED': ['GREEN'],
        'GREEN': ['YELLOW', 'RED'],
        'YELLOW': ['RED']
    }

    def __init__(self, initial_states=[None, 'RED', None], confirm_frames=3, strength_threshold=15):
        self.states = initial_states.copy()
        self.candiate_states = initial_states.copy()
        self.states_frame_count = [0, 0, 0]
        self.confirm_frames = confirm_frames
        self.strength_threshold = strength_threshold
        self.last_change_frames = [0, 0, 0]

    def update(self, candidates: list, frame_idx):
        print(
        f"\r{self.states}",
        end="",
        flush=True,
    )
        for i in range(3):
            if candidates[i] is None:
                continue
            else:
                candidate, strength = candidates[i]

                if strength < self.strength_threshold:
                    continue

                if candidate == self.states[i]:
                    self.states_frame_count[i] = 0
                    self.candiate_states[i] = None
                    continue

                if candidate == self.candiate_states[i]:
                    self.states_frame_count[i] += 1
                else:
                    self.candiate_states[i] = candidate
                    self.states_frame_count[i] = 1

                if self.states_frame_count[i] >= self.confirm_frames:
                    if candidate in self.ALLOWED_NEXT_STATES[self.states[i]]:
                        self.states[i] = candidate
                        self.last_change_frames[i] = frame_idx
                    self.states_frame_count[i] = 0

        return self.states
    
    def get_states(self):
        return self.states