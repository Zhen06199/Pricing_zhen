from math import gamma
import numpy as np
import random
import itertools
from Customer import *
from Provider import *
class Market:
    def __init__(self):

        self.customer = {}
        self.provider = {}
        self.can_choose = None
        self.state_space = None
        self.transit_pro = None

        self.filename_c = "/customer_comparision.txt"
        self.filename_p = "/provider_comparison.txt"
        self.provide_state = None
        self.customer_state = None
        self.state_value = None


        self.init()

    def init(self):
        self.load_provider()
        self.load_customer()
        self.load_resource_utility()
        self.initial_probabilities()
        self.choice_probabilities()
        self.init_state()
        self.state_space1()
        self.update_transit_probility()

    def load_provider(self):
        with open(self.filename_p,"r") as f:
            next(f)
            for line in f:
                fields = line.strip().split(",")
                ID = int(fields[0])
                supply = int(fields[1])
                energy_c = int(fields[2])
                latency_c = float(fields[3])
                reliability_c = float(fields[4])
                cost = float(fields[5])

                self.provider[ID] = Provider(ID,supply,energy_c,latency_c,reliability_c,cost)

    def load_customer(self):
        with open(self.filename_c, "r") as f:
            next(f)
            for line in f:
                fields = line.strip().split(",")
                ID = int(fields[0])
                capacity_demand = int(fields[1])
                energy_d = int(fields[2])
                latency_d = float(fields[3])
                reliability_d = float(fields[4])
                budget = float(fields[5])

                self.customer[ID] = Customer(ID, capacity_demand, energy_d, latency_d, reliability_d, budget)
        for p in self.provider:
            self.provider[p].choice_possibility = np.zeros(len(self.customer))

    def load_resource_utility(self):
        for c in self.customer:
            for p in self.provider:
                cur_p =self.provider[p]
                cur_match, cur_uti = self.customer[c].utility_r(cur_p)
                self.customer[c].match.append(cur_match)
                if cur_match:
                    self.customer[c].can_choose.append(p)
                self.customer[c].utility_resource.append(cur_uti)

    def initial_probabilities(self):
        num_p = len(self.provider)
        for c in self.customer:
            self.customer[c].choice_probabilities = np.ones(num_p)/num_p

    def choice_probabilities(self):
        for c in self.customer:
            cur_c = self.customer[c]
            total_uti = np.zeros(len(self.provider))
            for p in self.provider:
                cur_p = self.provider[p]
                if cur_c.match[p]:
                    total_uti[p] = -cur_c.capacity_demand * cur_p.price_policy + cur_c.utility_resource[p]
                else:
                    total_uti[p] = None
            exp_uti = np.exp(total_uti)
            exp_uti = np.where(np.isnan(exp_uti), 0, exp_uti)
            prob = exp_uti / np.sum(exp_uti)
            self.customer[c].choice_probabilities = exp_uti / np.sum(exp_uti)
            for p in self.provider:
                self.provider[p].choice_possibility[c] = self.customer[c].choice_probabilities[p]

        return 0

    def init_state(self):
        state_customer = []
        for c in self.customer:
            cur_c = self.customer[c]
            sample_ID = random.choices(range(len(cur_c.choice_probabilities)), weights=cur_c.choice_probabilities, k=1)[0]
            impossible_ID = [index for index, value in enumerate(cur_c.choice_probabilities) if value == 0]
            state_customer.append(sample_ID)
            for p in self.provider:
                for id in impossible_ID:
                    if p == id:
                        self.provider[p].not_match.append(c)

        for p in self.provider:
            self.provider[p].state = np.zeros(len(self.customer))
            positions = [index for index, value in enumerate(state_customer) if value == p]
            if positions:
                for num in positions:
                    self.provider[p].state[num] = 1
        self.customer_state = np.array(state_customer)
        return  0

    def update_state(self):
        state_customer = []
        for c in self.customer:
            cur_c = self.customer[c]
            sample_ID = random.choices(range(len(cur_c.choice_probabilities)), weights=cur_c.choice_probabilities, k=1)[
                0]
            state_customer.append(sample_ID)
        self.customer_state = state_customer


    def state_space1(self):
        self.can_choose = [self.customer[c].can_choose for c in self.customer]
        self.state_space = np.array(list(itertools.product(*self.can_choose)))

        for c in self.provider:
            self.provider[c].state_v = np.zeros(len(self.state_space))


        return 0

    def update_transit_probility(self):
        choice_pos = np.array([self.customer[c].choice_probabilities for c in self.customer])
        chosen_probs = choice_pos[np.arange(choice_pos.shape[0]), self.state_space]
        self.transit_pro = np.prod(chosen_probs, axis=1)

        return 0

    # formula (20) --> Ri(w,p)
    def revenue_function(self,price,state, provider_ID):
        customer_ID = [i for i, p in enumerate(state) if provider_ID == i]
        demand = np.array([self.customer[c].capacity_demand for c in customer_ID])
        revenue = demand * price
        return revenue

    # formula (20) --> P(w'|w,pi,p-i)
    def transit_prob_price(self,price, provider_ID):
        demand = np.array([self.customer[c].capacity_demand for c in self.customer])
        resource_uti = np.array([self.customer[c].utility_resource for c in self.customer])
        state_customer_prob = []
        for c in self.customer:
            cur_c = self.customer[c]
            total_uti = np.zeros(len(self.provider))
            for p in self.provider:
                cur_p = self.provider[p]
                if p == provider_ID:
                    if cur_c.match[p]:
                        total_uti[p] = -cur_c.capacity_demand * price + cur_c.utility_resource[p]
                    else:
                        total_uti[p] = None
                else:
                    if cur_c.match[p]:
                        total_uti[p] = -cur_c.capacity_demand * cur_p.price_policy + cur_c.utility_resource[p]
                    else:
                        total_uti[p] = None
            exp_uti = np.exp(total_uti)
            exp_uti = np.where(np.isnan(exp_uti), 0, exp_uti)
            choice_probabilities = exp_uti / np.sum(exp_uti)
            state_customer_prob.append(choice_probabilities)
        choice_pos =np.array(state_customer_prob)
        chosen_probs = choice_pos[np.arange(choice_pos.shape[0]), self.state_space]
        transit_pro = np.prod(chosen_probs, axis=1)

        return transit_pro

    def State_value(self,price, state,provider_ID, gamma):
        Revenue = self.revenue_function(price, state,provider_ID)
        P_transit = self.transit_prob_price(price, provider_ID)
        last_state_value = np.array(self.provider[provider_ID].state_v)
        V_cur = Revenue + gamma * np.dot(P_transit, last_state_value)

        return V_cur

    #这个是beta的变化，但是弃用
    def get_next_state(self):
        for p in self.provider:
            cur_p = self.provider[p]
            self.provider[p].next_state = []
            self.provider[p].state_possibility = []
            for i in range(len(cur_p.state)):
                if i not in cur_p.not_match:
                    # 拷贝原数组
                    new_array = cur_p.state.copy()
                    # 翻转第i位
                    new_array[i] = 1 - new_array[i]  # 0变1，1变0
                    # 添加到可能性列表
                    self.provider[p].next_state.append(new_array)

                    # 下一个状态的转移方程
                    diff = np.sum((new_array - cur_p.state) ** 2)
                    weighted_factor = np.exp(-diff)
                    #weighted_factor =1
                    multi = new_array * self.provider[p].choice_possibility + (1-new_array)*(1-self.provider[p].choice_possibility)
                    state_possibility = weighted_factor * np.prod(multi)
                    self.provider[p].state_possibility.append(state_possibility)


            self.provider[p].next_state.append(cur_p.state)
            multi = cur_p.state * self.provider[p].choice_possibility + (1 - cur_p.state) * (
                        1 - self.provider[p].choice_possibility)
            state_possibility = np.prod(multi)
            self.provider[p].state_possibility.append(state_possibility)
        return 0

def main():
    M1=Market()
    t = 0
    num_p = len(M1.provider)
    gamma = 0.95
    num_c = len(M1.customer)
    state_space = M1.state_space
    stop = False
    current_state = M1.customer_state
    price = []

    while not stop :
        t = t +1
        for i in range(num_p):
            V_i2s = []
            for state in state_space:
                V_max = 0
                P_max = 0
                for price in np.arange(0.2, 0.51, 0.01):  # 生成 1.0 到 2.0，步长 0.1
                    price = round(price, 2)
                    V_t = M1.State_value(price,state,i,gamma)
                    if V_max <= V_t:
                        V_max = V_t
                        P_max = price
                V_i2s.append(V_max)

                if state.all() == current_state.all():
                    M1.provider[i].price_policy = P_max
            M1.provider[i].state_v = V_i2s
        M1.update_transit_probility()
        M1.choice_probabilities() #update choice probability
        M1.update_state() #
        current_state = np.array(M1.customer_state)


        stop = 0



    # transit_pro=M1.transit_prob_price(1.1,3)
    state = M1.customer_state
    V = M1.State_value(1.1,state,1,0.95)
    return 0

if __name__ == '__main__':
    main()

