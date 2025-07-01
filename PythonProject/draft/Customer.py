from Provider import *
class Customer:
    def __init__(self, ID, capacity_demand, energy_d, latency_d, reliability_d, budget):
        self.ID = ID
        self.budget = budget
        self.capacity_demand = capacity_demand
        self.resource_demand = [energy_d, latency_d, reliability_d]
        self.match = []
        self.can_choose = []
        self.utility_resource = []
        self.choice_probabilities = []



    def utility_r(self,provider):
        #energy utility, if match
        energy_flag = self.resource_demand[0] == provider.resource_capacity[0]
        energy_uti = 0 if energy_flag else -1

        # latency utility, k is slope
        k = 1
        latency_demand = self.resource_demand[1]
        diff_l = latency_demand - provider.resource_capacity[1]
        latency_flag = diff_l>=0
        latency_uti = k*diff_l if latency_flag else -1

        # reliability utility,  q is slope
        q = 1
        reliability_demand = self.resource_demand[2]
        diff_r = provider.resource_capacity[2] - reliability_demand
        reliability_flag = diff_r >= 0
        reliability_uti = q * diff_r if reliability_flag else -1

        # total utility
        match = energy_flag & latency_flag & reliability_flag
        util_r = energy_uti+latency_uti+reliability_uti if match else -1

        return match, util_r





#---------------------------------------------------------------------

def main():
    print("Program is running")
    C1 = Customer(1,0,1,5,1,1)
    P1 = Provider(1,1,2,2)
    P2 = Provider(2, 0, 2, 2)

    a,b = C1.utility_r(P1)
    C1.utility_resource.append(b)
    c,d = C1.utility_r(P2)
    C1.utility_resource.append(d)

if __name__ == '__main__':
    main()