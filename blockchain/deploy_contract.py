from web3 import Web3
from solcx import compile_source, install_solc, set_solc_version

# install and set solidity compiler
install_solc("0.8.0")
set_solc_version("0.8.0")

# connect to ganache
web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))

# read contract
with open("blockchain/contract.sol", "r") as file:
    contract_source_code = file.read()

# compile contract
compiled_sol = compile_source(contract_source_code)

contract_interface = compiled_sol.popitem()[1]

# deploy contract
account = web3.eth.accounts[0]

contract = web3.eth.contract(
    abi=contract_interface["abi"],
    bytecode=contract_interface["bin"]
)

tx_hash = contract.constructor().transact({'from': account})
tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

print("Contract deployed at:", tx_receipt.contractAddress)