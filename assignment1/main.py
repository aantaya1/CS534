import copy

dirty = "dirty"
clean = "clean"

roomConfigurations = [[dirty, dirty],
                      [dirty, clean],
                      [clean, dirty],
                      [clean, clean]]


class SimpleReflexVacuumAgent:

    def __init__(self):
        self.curr_score = 2000

    def run_configurations(self):
        for j in range(0, 2):
            room = "left" if j is 0 else "right"

            for i in range(0, len(roomConfigurations)):

                curr_config = copy.deepcopy(roomConfigurations[i])
                print("Initial Room: " + room)
                print(roomConfigurations[i])

                while True:
                    if self.is_goal_state(curr_config):
                        break

                    if room is "left":
                        action = self.agent_function(room, curr_config[0])
                    else:
                        action = self.agent_function(room, curr_config[1])

                    print("Action:" + action)

                    if action is "suck" and room is "left":
                        curr_config[0] = "clean"
                        continue
                    elif action is "suck" and room is "right":
                        curr_config[1] = "clean"
                        continue

                    room = action

                print("Score: ", self.curr_score)
                self.curr_score = 2000

    def agent_function(self, room, cleanliness):

        if cleanliness is "dirty":
            return "suck"

        if room is "left":
            return "right"
        else:
            return "left"

    def is_goal_state(self, curr_state):
        if curr_state[0] is "clean" and curr_state[1] is "clean":
            return True
        else:
            self.curr_score = self.curr_score - 1
            return False


def main():
    agent = SimpleReflexVacuumAgent()
    agent.run_configurations()


main()
