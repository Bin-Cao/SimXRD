"""
Contact Information:
Mr. Cao Bin Email: bcao686@connect.hkust-gz.edu.cn
"""

from ase.db import connect
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Directory paths for the database files
traindb = ['./train1.db', './train2.db', './train3.db', './train4.db', './train5.db', './train6.db']
traindb_time = [5, 5, 5, 5, 5, 5]  # Simulation times for each training database
valdb = ['./val.db']
valdb_time = [1]  # Simulation times for validation database
testdb = ['./test.db']
testdb_time = [2]  # Simulation times for test database

split_ratio = [0.7, 0.1, 0.2]  # Ratio for splitting the data into training, validation, and test sets

def OLdataset(traindb, valdb, testdb, traindb_time, valdb_time, testdb_time, split_ratio):
    # Combine all database paths and simulation times into single lists
    ILdataset = traindb + valdb + testdb
    envtims = traindb_time + valdb_time + testdb_time
    
    if len(ILdataset) != len(envtims):
        raise ValueError('The number of database files should match the number of simulation times.')

    total_entries = 119569
    train_num = int(total_entries * split_ratio[0])
    val_num = int(total_entries * split_ratio[1])
    test_num = total_entries - val_num - train_num

    OLtraindb = connect('./OLtraindb')
    OLvaldb = connect('./OLvaldb')
    OLtestdb = connect('./OLtestdb')

    def process_database(index):
        dbfile = connect(ILdataset[index])
        print(f'Writing in the file {ILdataset[index]}')

        # Write training data
        for entry_id in tqdm(range(1, train_num * envtims[index] + 1), desc=f'Training {ILdataset[index]}'):
            entry = dbfile.get(id=entry_id)
            OLtraindb.write(
                atoms=entry.toatoms(),
                latt_dis=entry.latt_dis,
                intensity=entry.intensity,
                tager=entry.tager,
                chem_form=entry.chem_form,
                simulation_param=entry.simulation_param
            )

        # Write validation data
        for entry_id in tqdm(range(train_num * envtims[index] + 1, train_num * envtims[index] + val_num * envtims[index] + 1), desc=f'Validation {ILdataset[index]}'):
            entry = dbfile.get(id=entry_id)
            OLvaldb.write(
                atoms=entry.toatoms(),
                latt_dis=entry.latt_dis,
                intensity=entry.intensity,
                tager=entry.tager,
                chem_form=entry.chem_form,
                simulation_param=entry.simulation_param
            )

        # Write test data
        for entry_id in tqdm(range(train_num * envtims[index] + val_num * envtims[index] + 1, train_num * envtims[index] + val_num * envtims[index] + test_num * envtims[index] + 1), desc=f'Testing {ILdataset[index]}'):
            entry = dbfile.get(id=entry_id)
            OLtestdb.write(
                atoms=entry.toatoms(),
                latt_dis=entry.latt_dis,
                intensity=entry.intensity,
                tager=entry.tager,
                chem_form=entry.chem_form,
                simulation_param=entry.simulation_param
            )

    # Use ThreadPoolExecutor to parallelize the processing of databases
    with ThreadPoolExecutor(max_workers=len(ILdataset)) as executor:
        executor.map(process_database, range(len(ILdataset)))

    return True

if __name__ == '__main__':
    OLdataset(traindb, valdb, testdb, traindb_time, valdb_time, testdb_time, split_ratio)
