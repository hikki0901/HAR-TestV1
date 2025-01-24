class Frame:
    """Class to hold information about each frame"""

    def __init__(self, id, diff):
        self.id = id
        self.diff = diff

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return self.id > other.id

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

    def getMAXdiff(self, list=[]):
        """
        Find the max_diff_frame in the window
        """
        max_diff_frame = max(list, key=lambda frame: frame.diff, default=None)
        return max_diff_frame

    def find_possible_frame(self, list_frames):
        """
        Detect the possible frame
        """
        possible_frame = []
        window_frame = []
        window_size = 30
        m_suddenJudge = 3
        m_MinLengthOfShot = 8
        start_id_spot = [0]
        end_id_spot = []

        length = len(list_frames)
        index = 0
        while index < length:
            frame_item = list_frames[index]
            window_frame.append(frame_item)
            if len(window_frame) < window_size:
                index += 1
                if index == length - 1:
                    window_frame.append(list_frames[index])
                else:
                    continue

            # Find the max_diff_frame in the current window
            max_diff_frame = self.getMAXdiff(window_frame)

            if len(possible_frame) == 0:
                possible_frame.append(max_diff_frame)
                continue

            last_max_frame = possible_frame[-1]

            # Check if the difference is significant
            sum_start_id = last_max_frame.id + 1
            sum_end_id = max_diff_frame.id - 1

            id_no = sum_start_id
            sum_diff = 0
            while id_no <= sum_end_id:
                sum_frame_item = list_frames[id_no]
                sum_diff += sum_frame_item.diff
                id_no += 1

            average_diff = sum_diff / (sum_end_id - sum_start_id + 1)
            if max_diff_frame.diff >= (m_suddenJudge * average_diff):
                possible_frame.append(max_diff_frame)
                window_frame = []
                index = possible_frame[-1].id + m_MinLengthOfShot
                continue
            else:
                index = max_diff_frame.id + 1
                window_frame = []
                continue

        # Add the last frame if necessary
        if possible_frame:
            last_frame = list_frames[-1]
            if possible_frame[-1].id < last_frame.id:
                possible_frame.append(last_frame)

        # Ensure only one keyframe (the one with the highest diff)
        if possible_frame:
            single_keyframe = max(possible_frame, key=lambda frame: frame.diff)
            possible_frame = [single_keyframe]

        return possible_frame, start_id_spot, end_id_spot

    def optimize_frame(self, tag_frames, list_frames):
        '''
        Optimize the possible frame
        '''
        new_tag_frames = []
        frame_count = 10
        diff_threshold = 10
        diff_optimize = 2
        start_id_spot = [0]
        end_id_spot = []

        for tag_frame in tag_frames:
            tag_id = tag_frame.id

            # Check whether the difference of the possible frame is no less than 10
            if tag_frame.diff < diff_threshold:
                continue

            # Get the previous 10 frames
            pre_start_id = max(0, tag_id - frame_count)
            pre_end_id = tag_id - 1
            pre_sum_diff = sum(frame.diff for frame in list_frames[pre_start_id:pre_end_id + 1])

            # Get the subsequent 10 frames
            back_start_id = tag_id + 1
            back_end_id = min(len(list_frames), tag_id + frame_count)
            back_sum_diff = sum(frame.diff for frame in list_frames[back_start_id:back_end_id])

            # Calculate the difference of the previous and subsequent frames
            sum_diff = pre_sum_diff + back_sum_diff
            average_diff = sum_diff / (frame_count * 2)

            # Check whether the requirement is met
            if tag_frame.diff > (diff_optimize * average_diff):
                new_tag_frames.append(tag_frame)

        # Handle empty new_tag_frames
        if not new_tag_frames:
            # If no tag frames are selected, return the last frame
            new_tag_frames = [list_frames[-1]]

        # Get the index of the first and last frame of a shot
        for i in range(len(new_tag_frames)):
            start_id_spot.append(new_tag_frames[i].id)
            end_id_spot.append(new_tag_frames[i].id - 1)

        last_frame = list_frames[-1]
        if new_tag_frames[-1].id < last_frame.id:
            new_tag_frames.append(last_frame)

        end_id_spot.append(new_tag_frames[-1].id)

        return new_tag_frames, start_id_spot, end_id_spot