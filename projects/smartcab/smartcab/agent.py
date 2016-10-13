import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, epsilon = 0.7, alpha = 0.6, gamma = 0.4):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # Initialize any additional variables here
        self.directions = [None, 'forward', 'left', 'right'] #possible actions to take
        self.Q = {} # dictionary with index made by tuble (state, action) and values of float
        
        # Defaults used for setting learning weights
        self.exploration_rate = epsilon # converges to the inverse of the total number of trips
        self.learning_rate = alpha # learning rate
        self.discount_factor = gamma # discount
        
        #Final analysis variables
        self.trips_rewards = [] #log of all trips rewards in a list
        self.passed_trips = 0  #number of passed trips

    def reset(self, destination=None):
        '''called before each trip'''
        self.planner.route_to(destination)
        
        # Prepare for a new trip; reset any variables here, if required
        self.trip_penalty = 0 #points lost for single trip
        self.trip_reward = 0  #points gained for a single trip

    def greedy_action(self, state):
        '''returns the action with the best known Q value for a given state.'''
        Qvals = [self.Q.get((state, action), 0.0) for action in self.directions]
        max_Qval = max(Qvals)
        if Qvals.count(max_Qval) > 1: #multiple existing states
            best = [i for i in range(len(self.directions)) if Qvals[i] == max_Qval] #create list of indexes
            direction_idx = random.choice(best) 
        else:
            direction_idx = Qvals.index(max_Qval) #get the index as qvals orders is the same of the matching directions
        return self.directions[direction_idx]

    def choose_action(self, state):
        '''Decide if we should take a random action or a greedy action.
           
           This random_factor value decreases for each of the completed trips.
           It is used for initial choice between greedy and random moves.
           
           The initial random_factor decreases as trips get completed
           so greedy actions are chosen after some initial moves
           It depends on the eploration_rate value'''
        random_factor = self.exploration_rate / (self.passed_trips + self.exploration_rate)

        if random.random() < random_factor: # Compare it with random 0..1 values
            action = random.choice(self.directions) # Random action for exploring the enviroment
        else:
            action =  self.greedy_action(state) #Pick the best value for Q - Greedy action
        return action

    def learnQ(self, prev_state, action, reward, next_state):
        '''Determine Q val and update Q table.sss
           From here we can see that we have a visibility on one state ahead,
           and the learning is based only on the current state and the Q table'''

        # For unexplored sets of states we get 0 Q value.
        # The learning rate and current reward will determine the final Q value
        prev_val = self.Q.get((prev_state, action), 0.0)
        next_val = max([self.Q.get((next_state, action), 0.0) for direction in self.directions])
        
        # Generates a discounted Q value to slightly update Q.
        learned_val =  reward + self.discount_factor * next_val
        # Even with unexplored states we multiply the learning rate with the reward
        
        #         last Q val + ( gamma           * utility of the next state)
        new_val = prev_val + (self.learning_rate * (learned_val - prev_val))
        
        # set the new Qval for the previous state
        self.Q[(prev_state, action)] = new_val

    def update(self, t):
        '''main loop called at the begining of each state'''
        inputs = self.env.sense(self) # Gather inputs from environment to create state
        deadline = self.env.get_deadline(self)
        self.next_waypoint = self.planner.next_waypoint()  # chosen move by route planner, also displayed by simulator

        # Update state
        '''Update state: Important variables for state are light, oncoming, and left, 
           right is not required as driver can turn right on red without penalty.'''
        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'])

        # Select action according to your policy
        action = self.choose_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Get the new state and waypoint after taking the action
        inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
        
        new_state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'])

        # Learn policy based on state, action, reward, new_state
        self.learnQ(self.state, action, reward, new_state)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)

        '''Logging for analysis of the agent.'''
        self.trip_reward += reward

        # Determine if the agent has reached the destination within the number of alloted steps.
        if(self.env.done == True):
            self.passed_trips += 1
            self.trips_rewards.append(self.trip_reward)
        elif(self.env.get_deadline(self) <= 0):
            self.trips_rewards.append(self.trip_reward)

def run():
    """Run the agent for a finite number of trips."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to tracks
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trips

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trips
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    pass_rate = a.passed_trips * 100.0 / len(a.trips_rewards)

    #Print stats after running a finite number of trips.
    print("Learning Rate: {}, Discount Factor: {}, exploration_rate {}, Pass Rate: {}%, Explored States: {}, Mean Reward: {}, Number of Trials: {}").format(
        a.learning_rate, a.discount_factor, a.exploration_rate, pass_rate, len(a.Q), sum(a.trips_rewards)/len(a.trips_rewards), len(a.trips_rewards))

if __name__ == '__main__':
    run()