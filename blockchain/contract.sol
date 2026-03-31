// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FakeNewsEvidence {

    // SINGLE STORAGE (OPTIMIZED)
    mapping(bytes32 => bytes32) private hashPrediction;

    event EvidenceStored(bytes32 indexed newsHash, bytes32 prediction);

    // -----------------------------
    // STORE EVIDENCE
    // -----------------------------
    function storeEvidence(bytes32 _newsHash, bytes32 _prediction) external {

        require(hashPrediction[_newsHash] == bytes32(0), "Hash exists");

        hashPrediction[_newsHash] = _prediction;

        emit EvidenceStored(_newsHash, _prediction);
    }

    // -----------------------------
    // VERIFY HASH
    // -----------------------------
    function verifyHash(bytes32 _newsHash) external view returns (bool) {
        return hashPrediction[_newsHash] != bytes32(0);
    }

    // -----------------------------
    // GET PREDICTION
    // -----------------------------
    function getPrediction(bytes32 _newsHash) external view returns (bytes32) {
        bytes32 prediction = hashPrediction[_newsHash];
        require(prediction != bytes32(0), "Not found");
        return prediction;
    }
}