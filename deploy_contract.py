from solcx import compile_source, install_solc
from web3 import Web3

# Install Solidity compiler
install_solc('0.8.0')

# Connect to Ganache
web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
account = web3.eth.accounts[0]

# Read contract
with open("blockchain/contract.sol", "r") as file:
    contract_source = file.read()

# Compile
compiled_sol = compile_source(contract_source, solc_version="0.8.0")
contract_id, contract_interface = compiled_sol.popitem()

bytecode = contract_interface['bin']
abi = contract_interface['abi']

# Deploy
FakeNews = web3.eth.contract(abi=abi, bytecode=bytecode)

tx_hash = FakeNews.constructor().transact({'from': account})
tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

print("\n CONTRACT DEPLOYED!")
print("Address:", tx_receipt.contractAddress)