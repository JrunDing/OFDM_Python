import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import commpy as cpy
import os
from math import log10


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class OFDM(object):
    """
    @brief：OFDM类
    @param：snrdb, k, p, channeltype='multipath', modulationtype='QPSK', pilotvalue=3 + 3j
    @return：object
    """

    def __init__(self, snrdb, k, p, channeltype='multipath', modulationtype='QPSK', pilotvalue=3 + 3j):
        # 信道参数
        self.SNRdb = snrdb  # SISO信道信噪比
        self.channel_type = channeltype  # 信道类型，可选multipath/awgn，multipath计算信道和信号卷积后加AWGN，awgn直接加awgn

        # 调制参数
        self.modulation_type = modulationtype  # 调制方式，可选BPSK、QPSK、8PSK、QAM16、QAM64
        self.m_map = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "QAM16": 4, "QAM64": 6}
        self.mu = self.m_map[self.modulation_type]  # 每个子载波对应bit

        # OFDM参数
        self.K = k  # 子载波数
        self.CP = self.K // 4  # 循环前缀CP数
        self.P = p  # 每个OFDM符号的导频数
        self.pilotValue = pilotvalue  # 导频值
        self.allCarriers = np.arange(self.K)  # 全部子载波索引 ([0, 1, ... K-1])
        self.pilotCarriers = self.allCarriers[::self.K // self.P]  # 导频索引，从0开始每隔K//P个子载波变成导频，那么最后子载波数K-P，导频数P
        self.pilotCarriers = np.hstack([self.pilotCarriers, np.array([self.allCarriers[-1]])])  # 为了方便信道估计，将最后一个子载波也作为导频
        self.P = self.P + 1  # 导频的数量也需要加1
        self.dataCarriers = np.delete(self.allCarriers, self.pilotCarriers)  # 数据子载波索引

        # 每个OFDM符号的有效bit，即 数据子载波数 × mu，其余比特为循环前缀和导频
        self.payloadBits_per_OFDM = len(self.dataCarriers) * self.mu


    #   @brief：串并转换
    #   @param：离散信源 bits
    #   @return：bits.reshape((len(dataCarriers), bps))
    def SP(self, bits):
        return bits.reshape(len(dataCarriers), bpe)

    #   @brief：并转串
    #   @param：bits
    #   @return：bits.reshape((-1,))
    def PS(self, bits):
        return bits.reshape((-1,))

    #   @brief：信号调制
    #   @param：bits
    #   @return：symbol
    def modulation(self, bits):
        if self.modulation_type == "QPSK":
            PSK4 = cpy.PSKModem(4)
            symbol = PSK4.modulate(bits)
            return symbol
        elif self.modulation_type == "QAM64":
            QAM64 = cpy.QAMModem(64)
            symbol = QAM64.modulate(bits)
            return symbol
        elif self.modulation_type == "QAM16":
            QAM16 = cpy.QAMModem(16)
            symbol = QAM16.modulate(bits)
            return symbol
        elif self.modulation_type == "8PSK":
            PSK8 = cpy.PSKModem(8)
            symbol = PSK8.modulate(bits)
            return symbol
        elif self.modulation_type == "BPSK":
            BPSK = cpy.PSKModem(2)
            symbol = BPSK.modulate(bits)
            return symbol

    #   @brief：解调
    #   @param：symbol
    #   @return：bits
    def demodulation(self, symbol):
        if self.modulation_type == "QPSK":
            PSK4 = cpy.PSKModem(4)
            bits = PSK4.demodulate(symbol, demod_type='hard')
            return bits
        elif self.modulation_type == "QAM64":
            QAM64 = cpy.QAMModem(64)
            bits = QAM64.demodulate(symbol, demod_type='hard')
            return bits
        elif self.modulation_type == "QAM16":
            QAM16 = cpy.QAMModem(16)
            bits = QAM16.demodulate(symbol, demod_type='hard')
            return bits
        elif self.modulation_type == "8PSK":
            PSK8 = cpy.PSKModem(8)
            bits = PSK8.demodulate(symbol, demod_type='hard')
            return bits
        elif self.modulation_type == "BPSK":
            BPSK = cpy.PSKModem(2)
            bits = BPSK.demodulate(symbol, demod_type='hard')
            return bits

    #   @brief：得频域内OFDM数据+导频
    #   @param：payload
    #   @return：symbol
    def OFDM_symbol(self, signalModulated):
        symbol = np.zeros(self.K, dtype=complex) # 子载波位置
        symbol[self.pilotCarriers] = self.pilotValue  # 在导频位置插入导频
        symbol[self.dataCarriers] = signalModulated  # 在数据位置插入数据
        return symbol

    #   @brief：IDFT
    #   @param：OFDM频域离散数据
    #   @return：时域离散信号
    def IDFT(self, OFDMData):
        return np.fft.ifft(OFDMData)

    #   @brief：添加循环前缀
    #   @param：OFDM_time
    #   @return：具有循环前缀的OFDM时域信号
    def addCP(self, OFDMTime):
        cp = OFDMTime[-self.CP:]
        return np.hstack([cp, OFDMTime])

    #   @brief：限幅
    #   @param：离散信号x，限幅范围CL
    #   @return：限幅后信号x_clipped
    def clipping (self , x, CL):
        sigma = np.sqrt(np.mean(np.square(np.abs(x))))  # 输入平方和取均值，再开方
        CL = CL*sigma  # 限幅CL = CL×sigma
        x_clipped = x
        clipped_idx = abs(x_clipped) > CL
        x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx]*CL),abs(x_clipped[clipped_idx]))
        return x_clipped

    #   @brief：计算峰均功率比
    #   @param：离散信号x
    #   @return：PAPR[dB]
    def PAPR(self, x):
        Power = np.abs(x)**2
        PeakP = np.max(Power)
        AvgP = np.mean(Power)
        PAPR_dB = 10*np.log10(PeakP/AvgP)
        return PAPR_dB

    #   @brief：AWGN信道
    #   @param：x_s, snrDB
    #   @return：x_s + noise, noise_pwr
    def add_awgn(self, x_s, snrDB):
        data_pwr = np.mean(abs(x_s**2))
        noise_pwr = data_pwr/(10**(snrDB/10))
        noise = 1/np.sqrt(2) * (np.random.randn(len(x_s)) + 1j *
                                np.random.randn(len(x_s))) * np.sqrt(noise_pwr)
        return x_s + noise, noise_pwr

    #   @brief：信道模型
    #   @param：in_signal, SNRdb, channel_type="awgn"
    #   @return：out_signal, noise_pwr
    def channel(self, in_signal, SNRdb, channel_type="awgn"):
        # 假定SISO无线信道模型采用一个三抽头，定义冲激响应的抽头系数为[1, 0, 0.3+0.3j]
        channelResponse = np.array([1, 0, 0.3+0.3j])  # 信道冲击响应
        if channel_type == "multipath":
            convolved = np.convolve(in_signal, channelResponse)
            out_signal, noise_pwr = self.add_awgn(convolved, SNRdb)
        elif channel_type == "awgn":
            out_signal, noise_pwr = self.add_awgn(in_signal, SNRdb)
        return out_signal, noise_pwr

    #   @brief：去循环前缀
    #   @param：signal
    #   @return：signal[CP:(CP+K)]
    def removeCP(self, signal):
        return signal[self.CP:(self.CP+self.K)]

    #   @brief：DFT
    #   @param：时域离散信号
    #   @return：频域离散信号
    def DFT(self, OFDMRxNoCP):
        return np.fft.fft(OFDMRxNoCP)


    #   @brief：采用内插方式，对信道进行估计   LS信道估计
    #   @param：OFDMDemod
    #   @return：Hest
    def channelEstimateLS(self, OFDMDemod):
        pilots = OFDMDemod[self.pilotCarriers]  # 取导频处的数据
        Hest_at_pilots = pilots / self.pilotValue  # LS信道估计s
        # 在导频载波之间进行插值以获得估计，然后利用插值估计得到数据下标处的信道响应
        Hest_abs = interpolate.interp1d(self.pilotCarriers, abs(
            Hest_at_pilots), kind='linear')(self.allCarriers)
        Hest_phase = interpolate.interp1d(self.pilotCarriers, np.angle(
            Hest_at_pilots), kind='linear')(self.allCarriers)
        Hest = Hest_abs * np.exp(1j*Hest_phase)
        return Hest


    #   @brief：MMSE信道估计  假设已知信道的时延功率谱，即时域信道矩阵h
    #   @param：OFDMDemod, pilot_idx ,SNR, pilotValue
    #   @return：HestMMSE
    '''
    def channelEstimateMMSE(OFDMDemod, pilotIdx ,SNR, pilotValue):
        pilots = OFDMDemod[pilotIdx] #  取出导频位置值
        hn_pilot_LS = pilots / pilotValue  # (9,)  LS信道估计

        hn = np.array(
            [-1. + 1.22464680e-16j, 0.83357867 - 9.46647260e-01j, 0. + 0.00000000e+00j, 1.02569932 + 5.22276692e-01j,
            0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0.69663835 + 9.66204296e-01j,
            0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0.66443826 + 5.86925110e-01j])
        hn = np.pad(hn, (0, 64 - len(hn)), 'constant', constant_values=(0, 0))  # hn后面补0，补后共有64个元素 (64,)
        hn_f = np.fft.fft(hn)  # len = 64
        rf_list = np.correlate(hn_f, hn_f, mode='full')  # len = 64*2-1 = 127   1 × 127
        idx_offset = int((len(rf_list) - 1) / 2)  # 63
        rf_list = rf_list / rf_list[idx_offset]  # 1 × 127

        idx_Rhp = np.arange(64).reshape(64, 1) - pilotIdx.reshape(1, -1)  # 64 × 9
        idx_Rpp = pilotIdx.reshape(-1, 1) - pilotIdx.reshape(1, -1)  # 9 × 9

        def get_rf(index):
            index = index + idx_offset
            return rf_list[index]
        get_rf = np.vectorize(get_rf) # 1 × 1
        Rhp = get_rf(idx_Rhp)  # 64 × 9
        Rpp = get_rf(idx_Rpp) + np.eye(len(pilotIdx)) / (10 ** (SNR * 0.1))  # 9 × 9

        HestMMSE = np.ones_like(OFDMDemod)  # (64,)
        HestMMSE = (Rhp @ np.linalg.inv(Rpp) @ hn_pilot_LS).reshape(64,)  # 64×9  乘  (9,)
        return HestMMSE
    '''

    #   @brief：MMSE信道估计  matlab代码改
    #   @param：OFDMDemod, pilotIdx ,SNR, pilotValue, channelImpRes, Nfft, P, Nps
    #   接收数据  导频索引  信噪比  导频值  信道脉冲响应  fft长度  导频数  导频间隔
    #   @return：H_MMSE
    def channelEstimateMMSE(self, OFDMDemod, pilotIdx ,SNR, pilotValue, channelImpRes, Nfft, P, Nps):
        h = channelImpRes  # 信道冲击响应
        pilots = OFDMDemod[pilotIdx] #  取出导频位置值
        H_tilde = pilots / pilotValue  # (9,)  LS信道估计

        snr = 10**(SNR*0.1)
        k = np.arange(0, len(h)) # 1*3
        hh = h@(np.conj(h).T) # 1*1
        tmp = h*np.conj(h)*k # 1*3
        r = sum(tmp)/hh # 1*1
        r2 = tmp@(k.T)/hh # 1*1
        tau_rms = np.sqrt(r2-r**2) # 1*1
        df = 1.0/Nfft
        j2pi_tau_df = 1j*2*3.1415926*tau_rms*df # 1*1
        K1 = np.tile(np.arange(0, Nfft).reshape(1, len(np.arange(0, Nfft))).T, (1, P))  # 64*9
        K2 = np.tile(np.arange(0, P).reshape(1, len(np.arange(0, P))), (Nfft, 1))  # 64*9
        rf = np.reciprocal(1+j2pi_tau_df*(K1-K2*Nps))
        K3 = np.tile(np.arange(0, P).reshape(1, len(np.arange(0, P))).T, (1, P))
        K4 = np.tile(np.arange(0, P).reshape(1, len(np.arange(0, P))), (P, 1))
        rf2 = np.reciprocal(1+j2pi_tau_df*Nps*(K3-K4))
        Rhp = rf
        Rpp = rf2+np.eye(len(H_tilde), len(H_tilde))/snr
        H_MMSE = np.transpose(Rhp@np.linalg.inv(Rpp)@H_tilde.T)
        return H_MMSE

    '''
    def channelEstimateMMSE(OFDMDemod, SNR):
        # Rh * (Rh + beta / snr * I) ^ -1 * HLS
        #C_response = np.array([1, 0, 0.3+0.3j]).reshape(-1, 1)  # 信道冲击响应
        #C_response_H = np.conj(C_response).T
        C_response = OFDMDemod.reshape(-1, 1)
        C_response_H = np.conj(C_response).T
        R_HH = np.matmul(C_response, C_response_H) # 信道相关矩阵  64*64
        snr = 10**(SNR*0.1)
        W = np.matmul(R_HH, np.linalg.inv((R_HH+(1/snr)*np.eye(64)))) # 加权矩阵 QPSK的beta为1   64*64
        HhatLS = channelEstimateLS(OFDMDemod)
        HhatLS = HhatLS.reshape(-1, 1)
        HhatLMMS = np.matmul(W, HhatLS) # LMMSE信道估计
        return HhatLMMS.squeeze() # 压缩为一维
    '''

    #   @brief：均衡器，消除信道干扰，去ISI
    #   @param：OFDMDemod, Hest
    #   @return：OFDMDemod / Hest
    def equalize(self, OFDMDemod, Hest):
        return OFDMDemod / Hest

    #   @brief：获得数据比特
    #   @param：equalized
    #   @return：equalized[dataCarriers]
    def get_payload(self, equalized):
        return equalized[self.dataCarriers]



    #   @brief：OFDM通信系统仿真
    #   @param：void
    #   @return：void
    def OFDMSystemSimulation(self):
        '''
        [1 0 1 1 0 1 ……]
        如果是size=(payloadBits_per_OFDM, 1)，则变成：
        [
         [1]，
         [0]，
         [0]，
         [1]，
          ……
            ]
         '''
        # 信源
        bits = np.random.binomial(n=1, p=0.5, size=(
        self.payloadBits_per_OFDM,))  # 信源产生随机信号0/1  产生1×payloadBits_per_OFDM的矩阵，每个元素取值0-n，均匀分布p
        print('TX bits:')
        print(bits)
        # 发送设备
        signalModulated = self.modulation(bits)  # 调制，SP，频域并行数据
        OFDMData = self.OFDM_symbol(signalModulated)  # OFDM信号  插入导频
        print('Complex on each subcarrier:')
        print(OFDMData)
        OFDMTime = self.IDFT(OFDMData)  # 频域转时域，PS 64个OFDM点=一个OFDM符号
        OFDMWithCP = self.addCP(OFDMTime)  # 加循环前缀

        # SISO无线信道
        OFDMTx = OFDMWithCP  # 待发送时域串行比特流
        OFDMRx = self.channel(OFDMTx, self.SNRdb, self.channel_type)[0]  # 通过信道，[0]看原函数，表示只取输出信号

        # 接收设备
        OFDMRxNoCP = self.removeCP(OFDMRx)  # 去循环前缀
        OFDMDemod = self.DFT(OFDMRxNoCP)  # 时域转频域，SP
        HestMMSE = self.channelEstimateMMSE(OFDMDemod, self.pilotCarriers, self.SNRdb, self.pilotValue, np.array([1, 0, 0.3 + 0.3j]), self.K, self.P,
                                       self.K / (self.P - 1))  # 信道估计 MMSE
        Hest = self.channelEstimateLS(OFDMDemod)  # 信道估计LS

        '''
        #  绘制信道估计结果  信道估计结果即在每一子载波频点处信道H的幅值
        plt.figure(1)
        h = np.array([1, 0, 0.3 + 0.3j])
        H = np.fft.fft(h, 64) # 64长度的fft变换
        plt.plot(allCarriers, abs(Hest), label='LS')
        plt.plot(allCarriers, abs(H), label='Real')
        plt.plot(allCarriers, abs(HestMMSE), label='MMSE')
        plt.legend()
        plt.show()
        '''

        equalizedHest = self.equalize(OFDMDemod, Hest)  # 利用估计的信道信息，进行均衡
        OFDMRxData = self.get_payload(equalizedHest)  # 获取数据位置的数据
        bitsRx = self.demodulation(OFDMRxData)  # 解调，PS

        # 信宿
        print('RX bits:')
        print(bitsRx)
        print("SNRdB:", self.SNRdb, "    BER：", np.sum(abs(bits - bitsRx)) / len(bits))


if __name__ == '__main__':
    ofdmsystem = OFDM(15, 64, 8) # 信噪比、子载波数、导频数
    ofdmsystem.OFDMSystemSimulation()

