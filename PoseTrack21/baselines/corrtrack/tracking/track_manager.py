from collections import deque

class TrackManager:

    instance = None
    inactive_patience=None

    @staticmethod
    def get_instance():
        if TrackManager.inactive_patience is None:
            assert False and "inactive patience is not set. Please class TrackManager.set_inactive_patience() first"
        if TrackManager.instance is None:
            TrackManager.instance = TrackManager()

        return TrackManager.instance

    @staticmethod
    def set_inactive_patience(num_frames):
        TrackManager.inactive_patience = num_frames

    @staticmethod
    def new_track(queries,
                  kpts,
                  ann_id,
                  features,
                  curr_frame,
                  num_kpts,
                  bbx,
                  reid_features):

        track = {
            'queries': queries,
            'kpts': kpts,
            'ann_id': [ann_id],
            'features': features,
            'curr_frame': curr_frame,
            'continue': 1,
            'num_kpts': num_kpts,
            'bbx': bbx,
            'reid_features': deque([reid_features]),
            'is_dead': 0,
            'count_inactive': 0,
            'tracked_consecutive_frames': 1
        }

        return track

    def __init__(self) -> None:
        self.__all_tracks__ = TrackList()

    def get_tracks(self):
        return self.__all_tracks__

    def reset(self):
        self.__all_tracks__ = TrackList()

    def get_inactive_tracks(self, min_track_length=1):
        return [(t_idx, t) for t_idx, t in enumerate(self.__all_tracks__)
                if t['continue'] == 0 and t['is_dead'] == 0 and len(t['ann_id']) >= min_track_length]

    def get_active_tracks(self):
        return [(t_idx, t) for t_idx, t in enumerate(self.__all_tracks__)
                if t['continue'] == 1 and t['is_dead'] == 0]

    def add(self, track):
        self.__all_tracks__.append(track)

    def update_track(self,
                     track_idx,
                     queries,
                     kpts,
                     ann_id,
                     features,
                     curr_frame,
                     num_kpts,
                     bbox,
                     reid_features):

        self.__all_tracks__[track_idx]['queries'] = queries
        self.__all_tracks__[track_idx]['kpts'] = kpts
        self.__all_tracks__[track_idx]['ann_id'].append(ann_id)
        self.__all_tracks__[track_idx]['features'] = features
        self.__all_tracks__[track_idx]['curr_frame'] = curr_frame
        self.__all_tracks__[track_idx]['num_kpts'] = num_kpts
        self.__all_tracks__[track_idx]['bbox'] = bbox
        self.__all_tracks__[track_idx]['reid_features'].append(reid_features)
        
        if len(self.__all_tracks__[track_idx]['reid_features']) > self.inactive_patience:
            self.__all_tracks__[track_idx]['reid_features'].popleft()

        self.__all_tracks__[track_idx]['continue'] = 1   # if track was inactive, set track to active
        self.__all_tracks__[track_idx]['count_inactive'] = 0

    def kill_short_tracks(self, min_track_len):
        for t in self.__all_tracks__:
            if t['continue'] == 0 and t['is_dead'] == 0:
                if len(t['ann_id']) < min_track_len:
                    t['is_dead'] = 1

    def increment_inactive_counter_and_kill(self):
        for t in self.__all_tracks__:
            if t['continue'] == 0 and t['is_dead'] == 0:
                t['count_inactive'] += 1
                if t['count_inactive'] > self.inactive_patience:
                    t['is_dead'] = 1

    def print_track_statistics(self):
        alive, inactive, dead = 0, 0, 0
        for t in self.get_tracks():
            if t['continue'] == 1:
                alive += 1
            elif t['continue'] == 0 and t['is_dead'] == 0:
                inactive += 1
            else:
                dead += 1

        print(f"alive: {alive}, inactive: {inactive}, dead: {dead}")

class TrackList(list):
    """
    proxy class for track list in corr_tracking
    """

    def __init__(self):
        super().__init__()

    def __getitem__(self, idx):
        track =  super().__getitem__(idx)

        # Schroedingers Track: raise error if track is dead and alive
        if track['continue'] == 1 and track['is_dead'] == 1:
            raise ValueError("A track can't be dead and alive")

        return track

    def append(self, item):
        assert isinstance(item, dict) and "Proxy only takes dict!"
        super().append(item)
