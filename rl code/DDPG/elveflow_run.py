#notes
#	retained module namespacing (by removing * imports)
#	made an elveflow API wrapper function for error handling
#	made an OB1 class: simplifies calls by not needing Instr_ID and Calib
#	removed all globals except 2: ob1 and pid_running
#	wait for daemonized threads for clean shutdown
#	miscellaneous cleaning/reorganization and stylistic changes
#		<80 characters per line (mostly)
#		tabs for indentation + spaces for alignment (disregarding PEP8)
#TODO
#	immediate
#		rename pid.py -> pid_.py (to avoid importing *)
#		run and debug
#		check if the extraneous paths/imports can be removed (as noted below)
#		clean cohen_coon_tuning() and pid_.py
#	intermediate
#		move the elveflow wrappers to a separate file
#		add multiple y-axes to live plotter
#	eventual
#		coordinate sensor reading done herein and by LivePlot.y1
#		PID loop: acquire all (acquire=1) & then read from memory (acquire=0)?
#		replace print() statements with a logger (error/warn/info/debug)

# --[ paths ]------------------------------------------------------------------
#path_to_c       = '/cygdrive/c'
import sys
path_to_c        = 'C:'
path_to_desktop  = f'{path_to_c}/Users/kornberg/Desktop'
path_to_code     = f'{path_to_desktop}/code'
path_to_elveflow = f'{path_to_desktop}/esi_software'
path_to_config   = f'{path_to_elveflow}/config'
path_to_sdk      = f'{path_to_elveflow}/ESI_V3_08_02/ESI_V3_08_02/SDK_V3_08_02'
path_to_sdk_py64 = f'{path_to_sdk}/DLL/Python/Python_64'
sys.path.append(f'{path_to_sdk_py64}/DLL64') #TODO: needed?
sys.path.append(path_to_sdk_py64) #path to Elveflow64.py
sys.path.append(path_to_code)     #path to liveplot.py, pid.py
print('finished appending paths')

# --[ imports ]----------------------------------------------------------------
import threading
import time
from   email.header import UTF8 #TODO: needed?
from   array import array       #TODO: needed?
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import Elveflow64
import liveplot
import pid_
import pandas
print('finished imports')

# --[ classes ]----------------------------------------------------------------
class OB1:
	name = '02062435'                             #obtained from NIMAX
	regs = [2, 0, 0, 0]                           #regulator types
	id   = None                                   #placeholder for c_int32()
	class Calib:
		def __init__(self, calib_dir, name):
			self.array    = None                  #calibration array
			self.length   = 1000                  #calibration array length
			self.dirname  = calib_dir             #$(dirname  calibration_data)
			self.basename = f'{name}.calib'       #$(basename calibration_data)
			self.path     = f'{self.dirname}/{self.basename}'.replace('/', '\\')
			return
		pass #for auto-indentation
	def __init__(self, name, regs):
		self.name  = name
		self.regs  = regs
		self.calib = self.Calib(path_to_config, name)
		return
	pass #for auto-indentation

# --[ globals ]----------------------------------------------------------------
pid_running  = True
last_reading = None
last_time    = None

# --[ elveflow API wrapper ]---------------------------------------------------
def run_elveflow_api(api_function, *args):
	"""
	Wrapper function to call an API function and check for errors.

	:param api_function: The API function to call.
	:param args: Arguments to pass to the API function.
	"""
	error = api_function(*args)
	if error != 0: #TODO: deinitialize instrument if necessary
		err_msg = (
			f"Error {error} occurred while calling "
			f"{api_function.__name__}({args})"
		)
		raise Exception(err_msg)
	return

# --[ elveflow API ]-----------------------------------------------------------
def init_ob1(ob1):
	Instr_ID = ctypes.c_int32()
	run_elveflow_api(
		Elveflow64.OB1_Initialization,           #API function name
		ob1.name.encode('ascii'),                #device name in ASCII
		ob1.regs[0],                             #regulator type (channel 1)
		ob1.regs[1],                             #regulator type (channel 2)
		ob1.regs[2],                             #regulator type (channel 3)
		ob1.regs[3],                             #regulator type (channel 4)
		ctypes.byref(Instr_ID)                          #modified by API function
	)
	return Instr_ID

def init_mfs(ob1):
	run_elveflow_api(
		Elveflow64.OB1_Add_Sens,                 #API function name
		ob1.id,                                  #OB1 instrument ID
		1,                                       #channel number (1–4)           (XXX)
		5,                                       #sensor model (5 → MFS4)
		1,                                       #sensor type  (0=analog, 1=digital)
		1,                                       #digital calibration            (XXX)
		7,                                       #digital resolution (9–16 bits) (XXX)
		0                                        #custom sensor voltage
	)
	return

def bound_max_pressure(new_pressure, max_pressure = 1000):
	if new_pressure > max_pressure:
		return max_pressure
	else:
		return new_pressure
	pass

def set_pressure(ob1, new_pressure):
	new_pressure = bound_max_pressure(new_pressure)
	run_elveflow_api(
		Elveflow64.OB1_Set_Press,                #API function name
		ob1.id,                                  #OB1 instrument ID
		1,                                       #regulator channel (1–4)
		float(new_pressure),                     #pressure (mbar)
		ctypes.byref(ob1.calib.array),                  #calibration array
		ob1.calib.length                         #calibration array length
	)
	return

def get_pressure(ob1):
	current_pressure = ctypes.c_double()
	run_elveflow_api(
		Elveflow64.OB1_Get_Press,                #API function name
		ob1.id,                                  #OB1 instrument ID
		1,                                       #regulator channel (1–4)
		1,                                       #acquire data? (1=true, 0=false)
		ctypes.byref(ob1.calib.array),                  #calibration array
		ctypes.byref(current_pressure),                 #modified by API function
		ob1.calib.length                         #calibration array length
	)
	return current_pressure.value                #type: float

def get_sens(ob1):
	current_reading = ctypes.c_double()
	run_elveflow_api(
		Elveflow64.OB1_Get_Sens_Data,            #API function name
		ob1.id,                                  #OB1 instrument ID
		1,                                       #regulator channel (1–4) (XXX)
		1,                                       #acquire data? (1=true, 0=false)
		ctypes.byref(current_reading)                   #modified by API function
	)
	return current_reading.value                 #type: float

def get_sens_(ob1=None):
	global last_reading, last_time
	if (ob1):
		last_reading = get_sens(ob1)
		last_time    = time.time()
	return (last_reading, last_time)

def deinit_ob1(ob1):
	set_pressure(ob1, 0)
	run_elveflow_api(
		Elveflow64.OB1_Destructor,               #API function name
		ob1.id                                   #OB1 instrument ID
	)
	return

def calibrate_ob1_default(ob1):
	run_elveflow_api(
		Elveflow64.Elveflow_Calibration_Default, #API function name
		ctypes.byref(ob1.calib.array),                  #modified by API function
		ob1.calib.length                         #calibration array length
	)
	return

def calibrate_ob1_load(ob1):
	run_elveflow_api(
		Elveflow64.Elveflow_Calibration_Load,    #API function name
		ob1.calib.path.encode('ascii'),          #path of calibration file
		ctypes.byref(ob1.calib.array),                  #modified by API function
		ob1.calib.length                         #calibration array length
	)
	return

def calibrate_ob1_new(ob1):
	run_elveflow_api(
		Elveflow64.OB1_Calib,                    #API function name
		ob1.calib.array,                         #modified by API function (XXX)
		ob1.calib.length                         #calibration array length
	)
	run_elveflow_api(
		Elveflow64.Elveflow_Calibration_Save,    #API function name
		ob1.calib.path.encode('ascii'),          #path of calibration file
		ctypes.byref(ob1.calib.array),                  #modified by API function
		ob1.calib.length                         #calibration array length
	)
	return

def calibrate_ob1(ob1):
	while True:
		answer=input('select calibration type (default, load, new): ')
		if   answer=='default':
			calibrate_ob1_default(ob1)
			break
		elif answer=='load':
			calibrate_ob1_load(ob1)
			break
		elif answer=='new':
			calibrate_ob1_new(ob1)
			print(f'calibration data saved in {ob1.calib.path.encode("ascii")}')
			break
	return

def pid_loop(pid, ob1, data):
	global pid_running
	time_point = time.time()
	while pid_running:
		(current_flow, current_time) = get_sens_(ob1)
		current_pres = get_pressure(ob1)
		new_pres = pid.update(current_flow, current_pres, current_time)
		set_pressure(ob1, new_pres)
		#TODO: clean the following
		data.kps.append(pid.PTerm)
		data.kis.append(pid.ITerm * pid.Ki)
		data.kds.append(pid.DTerm * pid.Kd)
		data.pressures.append(current_pres)
		data.flows.append(current_flow)
		new_time = time.time()
		data.dts.append(new_time - time_point)
		time_point = new_time
		time.sleep(pid.sample_time)
		pass #for auto-indentation
	return

def cli(pid, kp, ki, kd, setpoint):
	while True:
		print(f"Current k_p = {kp}, current k_i = {ki}, current k_d = {kd}, setpoint = {setpoint} uL/min")
		kp = input("k_p: ")
		try:
			float(kp)
		except:
			print(f'invalid input: {kp}')
			continue
		ki = input("k_i: ")
		try:
			float(ki)
		except:
			print(f'invalid input: {ki}')
			continue
		kd = input("k_d: ")
		try:
			float(kd)
		except:
			print(f'invalid input: {kd}')
			continue
		setpoint = input("setpoint (uL/min): ")
		try:
			float(setpoint)
		except:
			print(f'invalid input: {setpoint}')
			continue
		# setting of variables
		pid.clear()
		pid.Kp       = float(kp)
		pid.Ki       = float(ki)
		pid.Kd       = float(kd)
		pid.SetPoint = float(setpoint)
	return

#obj
#	apply a pressure pulse and record the resulting flow rate over time
#TODO
#	consider starting pressure
def pulse_chase(ob1, pressure, duration):
	flows     = []
	times     = []
	init_time = time.time()
	set_pressure(ob1, pressure)
	while True:
		curr_flow, curr_time = get_sens_(ob1)
		flows.append(curr_flow)
		times.append(curr_time)
		if curr_time - init_time >= duration:
			break
		pass #for auto-indentation
	tf = pandas.DataFrame([flows, times])
	tf.to_csv('flow_vs_time_plus_10')
	return

#obj
#	auto-tune a PID controller using the method of Cohen & Coon
#usage
#	k_c, t_i, t_d = cohen_coon_tuning(ob1, 20)
#	print(f'k_c: {k_c}, t_i: {t_i}, t_d: {t_d}')
def cohen_coon_tuning(ob1, step_size):
	initial_pressure = get_pressure(ob1)
	in_steady_state = False
	while not in_steady_state:
		init_flows = []
		init_times = []
		for i in range(15):
			init_flows.append(get_sens(ob1))
			init_times.append(time.time())
			time.sleep(0.005)

		_, __ = pid_.find_steady_state_info(init_flows, init_times, consecutive_readings=10)

		if _ != None:
			in_steady_state = True
			break

	print('starting cohen-coon autotune')
	initial_flow = get_sens(ob1)
	print(f'setting pressure to: {initial_pressure + step_size}')
	set_pressure(ob1, initial_pressure + step_size)
	times = []
	response = []
	init_time = time.time()
	for i in range(500):
		times.append(time.time() - init_time)
		response.append(get_sens(ob1))
		time.sleep(0.001)

	time_to_steady_state, steady_state_flow =  pid_.find_steady_state_info(response, times)
	print(f'time_to_steady_state: {time_to_steady_state}, steady_state_flow: {steady_state_flow}')

	if time_to_steady_state == None:
		print('no steady state found')

	time_delay = pid_.find_time_delay(response, times)
	print(f'time_delay: {time_delay}')
	time_constant = pid_.find_time_constant(response, times, time_delay, initial_flow - steady_state_flow)
	print(f'time_constant: {time_constant}')
	kp = (steady_state_flow - initial_flow)/(time_constant+time_delay)


	a = kp * time_delay / time_constant
	t = time_delay / (time_constant + time_delay)

	if a == 0:
		a = 0.001
	if t == 0:
		t = 0.001

	# Calculate the tuning parameters based on the Cohen-Coon method
	k_c = (1.35/a) * (1 + (0.18*t)/(1-t))
	t_i = time_delay * ((2.5-2*t)/(1-0.39*t))
	t_d = time_delay * ((0.37-0.37*t)/(1-0.81*t))

	# Return the tuning parameters
	return k_c, t_i, t_d

def plot_data(data):
	if not data:
		return
	fig, axarr = plt.subplots(3, 1, figsize=(10,15))
	# --[ plot 1 ]-------------------------------------------------------------
	axarr[0].plot(data.flows, 'tab:red', label='flow')
	axarr[0].set_ylabel('flow rate (μL/min)')
	ax2 = axarr[0].twinx()
	ax2.plot(data.kps, 'tab:blue', label='Pterm')
	ax2.plot(data.kis, 'tab:green', label='Iterm')
	ax2.plot(data.kds, 'tab:orange', label='Dterm')
	ax2.plot(data.dts, 'tab:brown', label='Σ{P,I,D}')
	ax2.set_ylabel('{P,I,D,Σ, dt} terms')
	plt.legend()
	# --[ plot 2 ]-------------------------------------------------------------
	axarr[1].plot(data.dts, 'black', label='delta_t')
	axarr[1].set_ylabel('delta time')
	# --[ plot 3 ]-------------------------------------------------------------
	axarr[2].plot(data.pressures, 'tab:red', label='flow')
	axarr[2].set_ylabel('pressure')

	# --[ show ]---------------------------------------------------------------
	fig.tight_layout()
	plt.show()
	return

def main():
	ob1 = OB1('02062435', [2,0,0,0])
	ob1.id = init_ob1(ob1)
	init_mfs(ob1)
	ob1.calib.array = (ctypes.c_double*ob1.calib.length)()
	calibrate_ob1_load(ob1)
	print("setup finished")
	#pulse_chase(ob1, 10, 30)
	#try:
	#	data = main2(ob1)
	#except:
	#	data = None
	#print('shutting down')
	#deinit_ob1(ob1)
	# plot_data(data)
	return ob1

def main2(ob1):
	# p = 1e-3  #for old form of PID which did not add the current_pressure
	# i = 1e-2  #""
	# d = 1e-3  #""
	p = 0.1531
	i = 0.3656
	d = 0.0103
	setpoint = 100

	#NB: since lists are mutable, changes made by daemon will be visible here
	class Data: #for the time being
		kps       = []
		kis       = []
		kds       = []
		pressures = []
		flows     = []
		dts       = []
		pass #for auto-indentation
	data = Data()

	print('starting PID')
	pid = pid_.PID(P=p, I=i, D=d, sample_time=1e-2)
	pid.SetPoint = setpoint  #desired flow rate
	pid_thread = threading.Thread(
		target = pid_loop,
		args   = (pid, ob1, data),
		daemon = True,
	)
	pid_thread.start()

	print('starting CLI')
	cli_thread = threading.Thread(
		target = cli,
		args   = (pid, p, i, d, setpoint),
		daemon = True,
	)
	cli_thread.start()

	print('starting plot')
	#lp = liveplot.LivePlot(y1=lambda: np.random.rand())
	lp = liveplot.LivePlot(y1=get_sens_)
	lp.start() #blocks further execution

	#stop:
	global pid_running
	pid_running = False #signal the other threads to stop
	pid_thread.join()

	return data

if __name__ == '__main__':
	main()
	pass #for auto-indentation

# vim: ts=4 sts=0 sw=0 noet
