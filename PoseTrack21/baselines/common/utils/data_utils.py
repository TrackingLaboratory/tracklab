def get_posetrack_eval_dummy():
    det = {
        'images': [],
        'annotations': [],
        # categories must be a list containing precisely one item, describing the person structure
        'categories': [
            {
                'name': 'person',
                'keypoints': ["nose",
                              "head_bottom",  # "left_eye",
                              "head_top",  # "right_eye",
                              "left_ear",  # will be left zeroed out
                              "right_ear",  # will be left zeroed out
                              "left_shoulder",
                              "right_shoulder",
                              "left_elbow",
                              "right_elbow",
                              "left_wrist",
                              "right_wrist",
                              "left_hip",
                              "right_hip",
                              "left_knee",
                              "right_knee",
                              "left_ankle",
                              "right_ankle",
                              ]
            }
        ]
    }

    return det

