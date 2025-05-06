import numpy as np

class Provider:
    def __init__(self,ID, supply,energy_c, latency_c, reliability_c,cost):
        self.ID = ID
        self.supply = supply
        self.resource_capacity = [energy_c, latency_c, reliability_c]
        self.price_policy = 0.2
        self.cost = cost
        self.not_match =[]
        self.state = None
        self.next_state = None
        self.choice_possibility = None
        self.state_v = None
        self.state_possibility = None  #这里的每个状态的可能性加起来不是1，因为由于dimension的原因，没有把所有的可能性都列出来

def main():
    print("Program is running")
    P1 = Provider(1,0,1,5)
    print(f"ID is {P1.ID} Energy is {P1.resource_capacity[0]}")


if __name__ == '__main__':
    main()