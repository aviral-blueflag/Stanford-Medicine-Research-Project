import numpy as np
import elveflow_run as elveflow

ob1 = elveflow.main()

mu, L, r = 1, 1, 1
desired = 5
noise = 0.005

def get_pressure(prevPressure, setP):
    #return prevPressure + (setP - prevPressure) * (1 - np.exp(-0.1))
    return np.float32(elveflow.get_pressure(ob1))
def get_flow(pressure):
   # return pressure * np.pi * r**4 / (8 * mu * L) * np.random.normal(1, noise)
    return np.float32(elveflow.get_sens(ob1))

def get_reward(flow):
    return -(desired - flow) ** 2

def reset():
    return np.array([0, 0, desired])

def new_state(action, pressure, flow):
    pressure = get_pressure(pressure, action)
    flow = get_flow(pressure)
    elveflow.set_pressure(ob1, pressure)

    return np.array([pressure, action[0], np.float32(desired - flow)], dtype=np.float32), flow
