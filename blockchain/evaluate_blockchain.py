import time
from web3 import Web3

ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

contract_address = "0xAd6A50d55eC08626E499DF293f34431a314694d8"

contract_abi = [
{
"inputs":[
{"internalType":"string","name":"_newsHash","type":"string"},
{"internalType":"string","name":"_prediction","type":"string"}
],
"name":"storeEvidence",
"outputs":[],
"stateMutability":"nonpayable",
"type":"function"
}
]

contract = web3.eth.contract(address=contract_address, abi=contract_abi)
account = web3.eth.accounts[0]

start = time.time()

tx_hash = contract.functions.storeEvidence("testhash","Fake").transact({
'from': account
})

web3.eth.wait_for_transaction_receipt(tx_hash)

end = time.time()

print("Transaction Latency:", end-start, "seconds")

receipt = web3.eth.get_transaction_receipt(tx_hash)

gas_used = receipt.gasUsed
gas_price = web3.eth.gas_price

transaction_cost = gas_used * gas_price

print("Gas Used:", gas_used)
print("Gas Price:", gas_price)
print("Transaction Cost:", transaction_cost)

