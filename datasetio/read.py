import definitions
import scipy.io as scio


def getRecordName(index,contaminated=True):
    if contaminated:
        return 'sim' + str(index) + definitions.CON_SUFFIX
    else:
        return 'sim' + str(index) + definitions.PURE_SUFFIX


def readRecord(index, contaminated=True):
    if contaminated:
        data = scio.loadmat(definitions.CONTAMINATED_EEG)
        return data[getRecordName(index, contaminated)]
    else:
        data = scio.loadmat(definitions.PURE_EEG)
        return data[getRecordName(index, contaminated)]
