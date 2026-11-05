import ctypes as ct
import numpy as np

                            ########################################################
                            ###      AlazarMeasurer.dll Functions setup          ###
                            ########################################################

class ATS9872_driver():
    def __init__(self):
        # Load the enhanced multithreaded AlazarMeasurer.dll
        self.AlazarMeasurerdll = ct.CDLL('C:\\Users\\Franscisco Rouxinol\\Desktop\\Medidas\\Denys\\jun25\\alazar\\dev_hub\\AlazarMeasurer.dll')
        # self.AlazarMeasurerdll = ct.CDLL('C:\\Users\\Franscisco Rouxinol\Developer\\MeasurementLFDQ\\src\\instruments\\AlazarMeasurer.dll')            

        self.Ar = None # Amplitude reference for multiple IQ measurements
        self.recordsPerBuffer = None
        self.buffersPerAcquisition = None

        # Setup basic alazarCapture function (returns single I/Q pair - mean of all acquisitions)
        self.AlazarMeasurerdll.alazarCapture.argtypes = [
            ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_double, 
            ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_bool, 
            ct.POINTER(ct.c_char), ct.c_bool, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)
        ]
        self.AlazarMeasurerdll.alazarCapture.restype = ct.c_int

        # Setup enhanced alazarCaptureEnhanced function 
        self.AlazarMeasurerdll.alazarCaptureEnhanced.argtypes = [
            ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_double, 
            ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_bool, 
            ct.POINTER(ct.c_char), ct.c_bool, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_bool
        ]
        self.AlazarMeasurerdll.alazarCaptureEnhanced.restype = ct.c_int

        # Setup alazarCaptureCompatible function
        self.AlazarMeasurerdll.alazarCaptureCompatible.argtypes = [
            ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_double, 
            ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_bool, 
            ct.POINTER(ct.c_char), ct.c_bool, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)
        ]
        self.AlazarMeasurerdll.alazarCaptureCompatible.restype = ct.c_int

        # Setup alazarGetMultipleTriggerIQ function
        self.AlazarMeasurerdll.alazarGetMultipleTriggerIQ.argtypes = [
            ct.c_uint32, ct.c_uint32, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), 
            ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_char)
        ]
        self.AlazarMeasurerdll.alazarGetMultipleTriggerIQ.restype = ct.c_int

        # Setup alazarGetTriggerIQ function
        self.AlazarMeasurerdll.alazarGetTriggerIQ.argtypes = [
            ct.c_uint32, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_char)
        ]
        self.AlazarMeasurerdll.alazarGetTriggerIQ.restype = ct.c_int

        # Setup alazarGetPatternedTriggerIQ function
        self.AlazarMeasurerdll.alazarGetPatternedTriggerIQ.argtypes = [
            ct.c_int, ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), 
            ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_char)
        ]
        self.AlazarMeasurerdll.alazarGetPatternedTriggerIQ.restype = ct.c_int

        # Setup threading control functions
        self.AlazarMeasurerdll.alazarSetThreadingMode.argtypes = [ct.c_bool, ct.POINTER(ct.c_char)]
        self.AlazarMeasurerdll.alazarSetThreadingMode.restype = ct.c_int

        self.AlazarMeasurerdll.alazarGetThreadingStatus.argtypes = [ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_char)]
        self.AlazarMeasurerdll.alazarGetThreadingStatus.restype = ct.c_int

        # Setup NPT mode function
        self.AlazarMeasurerdll.alazarCaptureNPT.argtypes = [
            ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_double, 
            ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_bool, 
            ct.POINTER(ct.c_char), ct.c_bool, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)
        ]
        self.AlazarMeasurerdll.alazarCaptureNPT.restype = ct.c_int

    def capture(self,
                preTriggerSamples, 
                postTriggerSamples, 
                recordsPerBuffer, 
                buffersPerAcquisition, 
                powerReference_dBm, 
                inputLoad = 50, 
                triggerLevel_volts = 0.5, 
                triggerRange_volts = 3.0, 
                waveformHeadCut = 100, 
                period = 14, 
                buffer_timeout_ms = 5000, 
                decimation_value = 1, 
                TTL = False, 
                save = False):
       
        preTrigS = ct.c_int(preTriggerSamples)
        postTrigS = ct.c_int(postTriggerSamples)
        
        nRecords =  ct.c_int(recordsPerBuffer)
        nBuffers =  ct.c_int(buffersPerAcquisition)
        trigLevel = ct.c_double(triggerLevel_volts)
        trigRange = ct.c_double(triggerRange_volts)
        headCut = ct.c_int(waveformHeadCut)
        per = ct.c_int(period)
        timeout_ms = ct.c_int(buffer_timeout_ms)
        decimation_value =  ct.c_int(decimation_value)
        setTTL = ct.c_bool(TTL)
        saveData = ct.c_bool(save)
        errorMessage = ct.create_string_buffer(80)

        I = ct.c_double()
        Q = ct.c_double()
        
        # Reference Amplitude
        powerReference = 10**(powerReference_dBm/10)*1e-3 # Watts
        Ar = np.sqrt(2*powerReference*inputLoad)

        
        result = self.AlazarMeasurerdll.alazarCapture(preTrigS,
                                        postTrigS,
                                        nRecords,
                                        nBuffers,
                                        trigLevel,
                                        trigRange,
                                        headCut,
                                        per,
                                        timeout_ms,
                                        decimation_value,
                                        saveData,
                                        errorMessage,
                                        setTTL,
                                        ct.byref(I),
                                        ct.byref(Q))
        
        if result != 0:
            raise RuntimeError("Error Code "+str(result)+". "+errorMessage.value.decode())
        
        return I.value/Ar,Q.value/Ar

    def capture_enhanced(self,
                        preTriggerSamples, 
                        postTriggerSamples, 
                        recordsPerBuffer, 
                        buffersPerAcquisition, 
                        powerReference_dBm, 
                        inputLoad = 50, 
                        triggerLevel_volts = 0.5, 
                        triggerRange_volts = 3.0, 
                        waveformHeadCut = 100, 
                        period = 14, 
                        buffer_timeout_ms = 5000, 
                        decimation_value = 1, 
                        TTL = False, 
                        save = False,
                        enableMultithreading = False):
        """
        Enhanced capture function that populates global state for individual trigger access.
        Mathematically equivalent to basic capture() but allows subsequent access to individual triggers.
        """
        preTrigS = ct.c_int(preTriggerSamples)
        postTrigS = ct.c_int(postTriggerSamples)
        nRecords = ct.c_int(recordsPerBuffer)
        nBuffers = ct.c_int(buffersPerAcquisition)
        trigLevel = ct.c_double(triggerLevel_volts)
        trigRange = ct.c_double(triggerRange_volts)
        headCut = ct.c_int(waveformHeadCut)
        per = ct.c_int(period)
        timeout_ms = ct.c_int(buffer_timeout_ms)
        decimation_value = ct.c_int(decimation_value)
        setTTL = ct.c_bool(TTL)
        saveData = ct.c_bool(save)
        enableThreading = ct.c_bool(enableMultithreading)
        errorMessage = ct.create_string_buffer(80)

        I = ct.c_double()
        Q = ct.c_double()
        
        # Reference Amplitude
        powerReference = 10**(powerReference_dBm/10)*1e-3 # Watts
        self.Ar = np.sqrt(2*powerReference*inputLoad)
        self.recordsPerBuffer = recordsPerBuffer
        self.buffersPerAcquisition = buffersPerAcquisition

        result = self.AlazarMeasurerdll.alazarCaptureEnhanced(
                                        preTrigS, postTrigS, nRecords, nBuffers,
                                        trigLevel, trigRange, headCut, per,
                                        timeout_ms, decimation_value, saveData,
                                        errorMessage, setTTL, ct.byref(I), ct.byref(Q),
                                        enableThreading)
        
        if result != 0:
            raise RuntimeError("Error Code "+str(result)+". "+errorMessage.value.decode())
        
        return I.value/self.Ar, Q.value/self.Ar

    def capture_compatible(self,
                          preTriggerSamples, 
                          postTriggerSamples, 
                          recordsPerBuffer, 
                          buffersPerAcquisition, 
                          powerReference_dBm, 
                          inputLoad = 50, 
                          triggerLevel_volts = 0.5, 
                          triggerRange_volts = 3.0, 
                          waveformHeadCut = 100, 
                          period = 14, 
                          buffer_timeout_ms = 5000, 
                          decimation_value = 1, 
                          TTL = False, 
                          save = False):
        """
        Backwards-compatible enhanced capture function.
        Uses enhanced capture with multithreading disabled for full compatibility.
        """
        preTrigS = ct.c_int(preTriggerSamples)
        postTrigS = ct.c_int(postTriggerSamples)
        nRecords = ct.c_int(recordsPerBuffer)
        nBuffers = ct.c_int(buffersPerAcquisition)
        trigLevel = ct.c_double(triggerLevel_volts)
        trigRange = ct.c_double(triggerRange_volts)
        headCut = ct.c_int(waveformHeadCut)
        per = ct.c_int(period)
        timeout_ms = ct.c_int(buffer_timeout_ms)
        decimation_value = ct.c_int(decimation_value)
        setTTL = ct.c_bool(TTL)
        saveData = ct.c_bool(save)
        errorMessage = ct.create_string_buffer(80)

        I = ct.c_double()
        Q = ct.c_double()
        
        # Reference Amplitude
        powerReference = 10**(powerReference_dBm/10)*1e-3 # Watts
        Ar = np.sqrt(2*powerReference*inputLoad)

        result = self.AlazarMeasurerdll.alazarCaptureCompatible(
                                        preTrigS, postTrigS, nRecords, nBuffers,
                                        trigLevel, trigRange, headCut, per,
                                        timeout_ms, decimation_value, saveData,
                                        errorMessage, setTTL, ct.byref(I), ct.byref(Q))
        
        if result != 0:
            raise RuntimeError("Error Code "+str(result)+". "+errorMessage.value.decode())
        
        return I.value/Ar, Q.value/Ar

    def capture_npt(self,
                   postTriggerSamples, 
                   recordsPerBuffer, 
                   buffersPerAcquisition, 
                   powerReference_dBm, 
                   inputLoad = 50, 
                   triggerLevel_volts = 0.5, 
                   triggerRange_volts = 3.0, 
                   waveformHeadCut = 100, 
                   period = 14, 
                   buffer_timeout_ms = 5000, 
                   decimation_value = 1, 
                   TTL = False, 
                   save = False):
        """
        NPT (No Pre-Trigger) mode capture function.
        """
        postTrigS = ct.c_int(postTriggerSamples)
        nRecords = ct.c_int(recordsPerBuffer)
        nBuffers = ct.c_int(buffersPerAcquisition)
        trigLevel = ct.c_double(triggerLevel_volts)
        trigRange = ct.c_double(triggerRange_volts)
        headCut = ct.c_int(waveformHeadCut)
        per = ct.c_int(period)
        timeout_ms = ct.c_int(buffer_timeout_ms)
        decimation_value = ct.c_int(decimation_value)
        setTTL = ct.c_bool(TTL)
        saveData = ct.c_bool(save)
        errorMessage = ct.create_string_buffer(80)

        I = ct.c_double()
        Q = ct.c_double()
        
        # Reference Amplitude
        powerReference = 10**(powerReference_dBm/10)*1e-3 # Watts
        Ar = np.sqrt(2*powerReference*inputLoad)

        result = self.AlazarMeasurerdll.alazarCaptureNPT(
                                        postTrigS, nRecords, nBuffers,
                                        trigLevel, trigRange, headCut, per,
                                        timeout_ms, decimation_value, saveData,
                                        errorMessage, setTTL, ct.byref(I), ct.byref(Q))
        
        if result != 0:
            raise RuntimeError("Error Code "+str(result)+". "+errorMessage.value.decode())
        
        return I.value/Ar, Q.value/Ar

    def get_trigger_iq(self, triggerIndex):
        """
        Get I/Q values for a specific trigger event.
        Must call capture_enhanced() or capture_compatible() first.
        """
        I_value = ct.c_double()
        Q_value = ct.c_double()
        errorMessage = ct.create_string_buffer(80)

        result = self.AlazarMeasurerdll.alazarGetTriggerIQ(
            ct.c_uint32(triggerIndex),
            ct.byref(I_value),
            ct.byref(Q_value),
            errorMessage
        )

        if result != 0:
            raise RuntimeError(f"Error retrieving trigger {triggerIndex} IQ values: {errorMessage.value.decode()}")
            
        return I_value.value/self.Ar, Q_value.value/self.Ar

    def get_multiple_trigger_iq(self, startIndex=0, count=None):
        """
        Get I/Q values for multiple trigger events.
        Must call capture_enhanced() or capture_compatible() first.
        """
        if count is None:
            count = 1000  # Large default, will be limited by actual triggers
            
        i_values = (ct.c_double * count)()
        q_values = (ct.c_double * count)()
        retrieved_count = ct.c_uint32(0)
        errorMessage = ct.create_string_buffer(80)

        result = self.AlazarMeasurerdll.alazarGetMultipleTriggerIQ(
            ct.c_uint32(startIndex),
            ct.c_uint32(count),
            i_values,
            q_values,
            ct.byref(retrieved_count),
            errorMessage
        )

        if result != 0:
            raise RuntimeError(f"Error retrieving multiple trigger IQ values: {errorMessage.value.decode()}")
            
        print(f"Retrieved {retrieved_count.value} trigger IQ values starting from index {startIndex}")

        i_np = np.array([i_values[i] for i in range(retrieved_count.value)], dtype=np.float64)
        q_np = np.array([q_values[i] for i in range(retrieved_count.value)], dtype=np.float64)

        return i_np/self.Ar, q_np/self.Ar

    def get_patterned_trigger_iq(self, patternIndex, patternSize, max_count=None):
        """
        Get I/Q values for a specific pattern position within a repeating pattern of triggers.
        Must call capture_enhanced() or capture_compatible() first.
        
        Parameters
        ----------
        patternIndex : int
            Which trigger type in the pattern (0, 1, etc.)
        patternSize : int
            Total number of triggers in pattern (e.g., 2 for alternating)
        max_count : int, optional
            Maximum number of values to retrieve. If None, allocates enough for all possible matches.
            
        Returns
        -------
        tuple(ndarray, ndarray)
            Arrays of I and Q values for the specified pattern position
        """
        # If max_count is not specified, estimate a reasonable size based on total trigger count
        # Assume roughly totalTriggers/patternSize matches
        if max_count is None:
            # This is just an estimate - we'll use the actual retrieved count later
            max_count = int((self.buffersPerAcquisition-1)*self.recordsPerBuffer)
        
        i_values = (ct.c_double * max_count)()
        q_values = (ct.c_double * max_count)()
        retrieved_count = ct.c_uint32(0)
        errorMessage = ct.create_string_buffer(1024)

        result = self.AlazarMeasurerdll.alazarGetPatternedTriggerIQ(
            ct.c_int(patternIndex),
            ct.c_int(patternSize),
            i_values,
            q_values,
            ct.byref(retrieved_count),
            errorMessage
        )

        if result != 0:
            error_msg = errorMessage.value.decode()
            print(f"Error retrieving patterned trigger IQ values: {error_msg}")
            return np.array([]), np.array([])
            
        print(f"Retrieved {retrieved_count.value} trigger IQ values for pattern index {patternIndex}/{patternSize}")

        # Convert to numpy arrays for easier manipulation
        i_np = np.array([i_values[i] for i in range(retrieved_count.value)], dtype=np.float64)
        q_np = np.array([q_values[i] for i in range(retrieved_count.value)], dtype=np.float64)

        # If Ar (amplitude reference) is set, normalize the values
        if self.Ar is not None:
            return i_np/self.Ar, q_np/self.Ar
        else:
            return i_np, q_np

    def capture_multiple_trigger_iq(self, records_per_buffer, buffers_per_acquisition, startIndex=0, n_pulses=None, error_msg=None):
        """
        Compatibility wrapper for the existing notebook functions.
        """
        total_triggers = records_per_buffer * buffers_per_acquisition
        return self.get_multiple_trigger_iq(startIndex, total_triggers)

    def set_threading_mode(self, enable_threading):
        """
        Enable or disable multithreading mode.
        """
        errorMessage = ct.create_string_buffer(1024)
        
        result = self.AlazarMeasurerdll.alazarSetThreadingMode(
            ct.c_bool(enable_threading),
            errorMessage
        )
        
        if result != 0:
            raise RuntimeError(f"Error setting threading mode: {errorMessage.value.decode()}")
            
        return errorMessage.value.decode()

    def get_threading_status(self):
        """
        Get current threading status.
        """
        thread_count = ct.c_uint32(0)
        statusMessage = ct.create_string_buffer(1024)
        
        result = self.AlazarMeasurerdll.alazarGetThreadingStatus(
            ct.byref(thread_count),
            statusMessage
        )
        
        if result != 0:
            raise RuntimeError("Error getting threading status")
            
        return thread_count.value, statusMessage.value.decode()