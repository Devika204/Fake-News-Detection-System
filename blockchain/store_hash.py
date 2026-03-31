from web3 import Web3
import hashlib

# -----------------------------
# CONNECT TO GANACHE
# -----------------------------
ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

if not web3.is_connected():
    raise Exception("❌ Web3 not connected")

# -----------------------------
# CONTRACT DETAILS (UPDATED ABI)
# -----------------------------
# contract_address = "0x857A4A8A9561fb0124791F8251Bf72eD3A0cEDF1"
contract_address = "0x857A4A8A9561fb0124791F8251Bf72eD3A0cEDF1"

contract_abi = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "_newsHash", "type": "bytes32"},
            {"internalType": "bytes32", "name": "_prediction", "type": "bytes32"}
        ],
        "name": "storeEvidence",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "_newsHash", "type": "bytes32"}
        ],
        "name": "verifyHash",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "_newsHash", "type": "bytes32"}
        ],
        "name": "getPrediction",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function"
    }
]

contract = web3.eth.contract(address=contract_address, abi=contract_abi)
account = web3.eth.accounts[0]

# -----------------------------
# HASH FUNCTIONS (FIXED)
# -----------------------------
def generate_hash(text):
    return Web3.keccak(text=text)

def generate_prediction_hash(label):
    return Web3.keccak(text=label)

# -----------------------------
# STORE ON BLOCKCHAIN
# -----------------------------
def store_hash_on_chain(news_hash, label):
    try:
        prediction_hash = generate_prediction_hash(label)

        tx_hash = contract.functions.storeEvidence(
            news_hash,
            prediction_hash
        ).transact({'from': account})

        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

        return receipt.transactionHash.hex()

    except Exception as e:
        raise Exception(f"Store failed: {e}")

# -----------------------------
# VERIFY HASH
# -----------------------------
def verify_hash(news_hash):
    try:
        return contract.functions.verifyHash(news_hash).call()
    except Exception as e:
        raise Exception(f"Verify failed: {e}")

# -----------------------------
# GET PREDICTION
# -----------------------------
def get_prediction(news_hash):
    try:
        result = contract.functions.getPrediction(news_hash).call()

        # Convert bytes32 → readable string (optional)
        return result.hex()

    except Exception as e:
        raise Exception(f"Get prediction failed: {e}")