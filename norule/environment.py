class Environment:
    def __init__(self, gridworld_size:Tuple, num_mine:int):

        self.gridworld_size = gridworld_size
        self.nrow, self.ncol = self.gridworld_size
        self.num_mine = num_mine

        # points == action space (index)
        self.points = np.arange(self.nrow * self.ncol)
        self.num_actions = len(self.points)

        # reward, done 딕셔너리
        self.reward_dict = {'mine':-1, 'empty':1, 'overlapped':-1, 'guess':0.3, 'clear':1}
        self.done_dict = {'mine':True, 'empty':False, 'overlapped':False, 'guess':False, 'clear':True}

        # 지뢰 랜덤으로 배정
        self.mine_points = np.random.choice(self.points, self.num_mine, replace=False)

        # 정답 맵
        self.map_answer, self.mine_bool = self.make_answer_map()

        # state
        self.present_state = np.full((self.nrow, self.ncol), -1)


    def make_answer_map(self):
        '''
        정답 맵을 생성하는 함수
        output: 정답 맵, 지뢰 여부를 담은 bool 맵
        '''
        answer_map = np.full(shape=(self.nrow, self.ncol), fill_value=0)
        x, y = np.divmod(self.mine_points, self.ncol)
        answer_map[x, y] = -2
        mine_bool = (answer_map==-2)

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for idx in self.points:
            if idx in self.mine_points:
                continue
            else:
                x, y = divmod(idx, self.ncol)
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.nrow and 0 <= ny < self.ncol:
                        if mine_bool[nx, ny]:
                            answer_map[x, y] += 1

        return answer_map, mine_bool


    def bfs_minesweeper(self, clicked_idx:int):
        '''
        현재 state에서 클릭할 좌표에 따라 맵을 열어주는 함수
        input : clicked_idx
        output : clicked_idx에 따라 열린 맵(array)
        '''
        act_x, act_y = divmod(clicked_idx, self.ncol)
        queue = deque([(act_x, act_y)])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]

        result = self.present_state.copy()

        while queue:
            x, y = queue.popleft()

            if result[x, y] != -1:
                continue

            result[x, y] = self.map_answer[x, y]

            if self.map_answer[x,y] == 0:
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.nrow and 0 <= ny < self.ncol and result[nx, ny] == -1:
                        queue.append((nx,ny)) # 좌표 -> 인덱스 역산

        return result


    def check_guess(self, clicked_idx:int):
        '''
        clicked_idx가 guess인지 확인하는 함수
        input : clicked_idx
        output : clicked_idx가 guess인지 (bool)
        clicked_idx 주변 8칸이 모두 열리지 않은 경우 guess
        '''
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
        x, y = divmod(clicked_idx, self.ncol)
        result = 0

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.nrow and 0 <= ny < self.ncol:
                if self.present_state[nx, ny] == -1:
                    result += 1

        if result == 8:
            return True
        else:
            return False


    def move_mine(self, action_idx:int):
        '''
        에이전트가 첫 번째로 선택한 action이 지뢰인 경우
        해당 좌표의 지뢰를 다른 곳으로 옮기는 함수
        input : action_idx
        '''
        empty_points = np.setdiff1d(self.points, self.mine_points)
        new_mine = np.random.choice(empty_points, 1)

        self.mine_points = np.delete(self.mine_points, np.where(self.mine_points == action_idx))
        self.mine_points = np.append(self.mine_points, new_mine[0])

        # map reset
        self.map_answer, self.mine_bool = self.make_answer_map()
        self.present_state = np.full((self.nrow, self.ncol), -1)


    def step(self, action_idx:int):
        '''
        에이전트가 선택한 action에 따라 주어지는 next_state, reward, done
        input : action_idx
        output : next_state, reward, done, clear
        '''
        x, y = divmod(action_idx, self.ncol)

        # 첫번째 action인 경우
        if np.sum(self.present_state != -1) == 0 :
            if action_idx in self.mine_points:
                # 만약 start 좌표에 지뢰가 있는 경우 옮기기
                self.move_mine(action_idx)

        next_state = self.bfs_minesweeper(action_idx)

        # ======
        # reward
        if action_idx in self.mine_points:
            # 지뢰
            reward = self.reward_dict['mine']
            done = self.done_dict['mine']
            clear = False

        elif np.sum(next_state == -1) == self.num_mine:
            # 클리어
            reward = self.reward_dict['clear']
            done = self.done_dict['clear']
            clear = True

        else :
            clear = False
            guess = self.check_guess(action_idx)

            if self.present_state[x,y] != -1:
                # 중복 행동
                reward = self.reward_dict['overlapped']
                done = self.done_dict['overlapped']

            elif guess:
                # 추측 행동
                reward = self.reward_dict['guess']
                done = self.done_dict['guess']

            else:
                # 좋은 행동
                reward = self.reward_dict['empty']
                done = self.done_dict['empty']

        # 현재 state 업데이트
        self.present_state = next_state

        return next_state, reward, done, clear


    def reset(self):
        '''
        환경을 리셋하는 함수
        지뢰를 랜덤 초기화, 정답 맵과 state 초기화
        '''
        self.mine_points = np.random.choice(self.points, self.num_mine, replace=False)
        self.map_answer, self.mine_bool = self.make_answer_map()
        self.present_state = np.full((self.nrow, self.ncol), -1)


    def render(self, state):
        '''
        입력받은 state를 df타입으로 render
        input: state
        output: rendered state display
        '''
        render_state = np.full(shape=(self.nrow, self.ncol), fill_value=".")

        for idx in self.points:
            x, y = divmod(idx, self.ncol)
            if state[x,y] == -1:
                continue
            elif state[x,y] == -2:
                render_state[x,y] = "M"
            else:
                render_state[x,y] = state[x,y]

        render_state = pd.DataFrame(render_state)
        render_state = render_state.style.applymap(self.render_color)
        display(render_state)


    def render_answer(self):
        '''
        answer map을 df타입으로 render
        output: rendered answer map display
        '''
        render_state = np.full(shape=(self.nrow, self.ncol), fill_value=".")

        for idx in self.points:
            x, y = divmod(idx, self.ncol)
            if self.map_answer[x,y] == -2:
                render_state[x,y] = "M"
            else:
                render_state[x,y] = str(self.map_answer[x,y])

        render_state = pd.DataFrame(render_state)
        render_state = render_state.style.applymap(self.render_color)
        display(render_state)


    def render_color(self, var):
        color = {'0':'black', '1':"skyblue", '2':'lightgreen', '3':'red', '4':'violet', '5':'brown',
                 '6':'turquoise', '7':'grey', '8':'black', 'M':'white', '.':'black'}
        return f"color: {color[var]}"


    def samples(self, num:int):
        '''
        입력받은 개수만큼의 지뢰 idx 리스트(특정 게임 판의 고유성) 생성
        input: num
        output: num 개수만큼의 지뢰 idx 리스트를 담은 array
        '''
        sample_mine_points = []

        for i in range(num):
            self.mine_points = np.random.choice(self.points, self.num_mine, replace=False)
            sample_mine_points.append(self.mine_points)

        return sample_mine_points

    def train_reset(self, samples:np.array):
        '''
        입력받은 샘플 게임판으로만 랜덤으로 리셋하는 함수
        input: samples
        '''
        self.mine_points = random.sample(samples, 1)[0]
        self.map_answer, self.mine_bool = self.make_answer_map()
        self.present_state = np.full((self.nrow, self.ncol), -1)


    def check_18_up(self, state):
        '''
        18개 이상 열린 state인지 확인하는 함수
        input: state
        output: 18개 이상 열린 state인지 여부 (bool)
        '''
        if np.sum(state != -1) >= 18:
            return True
        else:
            return False
