import time


# タイマーを定義したクラス
# you can use this class as timer
class Timer:
    def __init__(self):
        print('Timer started.')
        self.start = time.time()
        self.time_sum = 0
        self.time_sum_stopped = 0
        self.time_sum_minute = 0
        self.time_sum_minute_stopped = 0
        self.timer_status = True

    def time_elapsed(self):
        if not self.timer_status:
            print('being stopped : ', self.time_sum_minute_stopped, 'minute(s)')
            return 'Timer is now stopped.'
        self.time_sum += time.time() - self.start
        self.time_sum_minute = self.time_sum // 60
        self.start = time.time()
        print(self.time_sum, 'seconds elapsed.')
        print(int(self.time_sum_minute), ' minute(s) elapsed.')
        return self.time_sum

    def reset(self):
        self.__init__()

    def stop(self):
        if not self.timer_status:
            print('Timer already stopped.')
            return False
        self.time_sum_stopped = self.time_elapsed()
        self.time_sum_minute_stopped = self.time_sum_minute
        self.timer_status = False
        print('Timer stopped.')

    def restart(self):
        if self.timer_status:
            print('Timer now working.')
            return False
        self.time_sum = self.time_sum_stopped
        self.time_sum_minute = self.time_sum_minute_stopped
        self.start = time.time()
        self.timer_status = True
        print('Timer restarted.')
