#!/usr/bin/env python3

"""
Advanced Network Traffic Analyzer with Threat Detection
===================================================

This application provides real-time network traffic monitoring with advanced threat
detection capabilities using machine learning. It captures network packets,
analyzes them for potential security threats, and stores the results in
Elasticsearch for further analysis and visualization through Kibana.

Main Features:
- Real-time packet capture and analysis
- Machine learning-based anomaly detection
- Traffic pattern analysis
- Threat classification
- Integration with Elasticsearch for data storage
- Real-time alerting system
"""

import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from scapy.all import sniff, IP, TCP, UDP
from elasticsearch import Elasticsearch
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




class ThreatDetector:
    """
    Handles threat detection using machine learning algorithms.
    
    This class implements various threat detection techniques including anomaly
    detection and pattern recognition using machine learning models.
    
    Attributes:
        model (IsolationForest): Machine learning model for anomaly detection
        scaler (StandardScaler): Feature scaler for normalizing input data
    """
    
    def __init__(self):
        """Initialize the ThreatDetector with ML models and scalers."""
        self.model = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    @staticmethod
    def _extract_features(packet_data: Dict) -> List[float]:
        """
        Extract numerical features from packet data for ML analysis.
        
        Args:
            packet_data: Dictionary containing packet information
            
        Returns:
            List of numerical features for ML processing
        """
        features = [
            packet_data['length'],
            hash(packet_data['src_ip']) % 1000000,
            hash(packet_data['dst_ip']) % 1000000,
            packet_data['src_port'] if packet_data['src_port'] else 0,
            packet_data['dst_port'] if packet_data['dst_port'] else 0
        ]
        return features
    
    def train_model(self, historical_data: List[Dict]) -> None:
        """
        Train the anomaly detection model using historical network data.
        
        Args:
            historical_data: List of packet data dictionaries for training
        """
        features = [self._extract_features(packet) for packet in historical_data]
        scaled_features = self.scaler.fit_transform(features)
        self.model.fit(scaled_features)
        
    def detect_threats(self, packet_data: Dict) -> Tuple[bool, float]:
        """
        Analyze a single packet for potential threats.
        
        Args:
            packet_data: Dictionary containing packet information
            
        Returns:
            Tuple of (is_threat, anomaly_score)
        """
        features = self._extract_features(packet_data)
        scaled_features = self.scaler.transform([features])
        score = self.model.score_samples([scaled_features[0]])[0]
        is_threat = score < -0.5  # Threshold for anomaly detection
        
        return is_threat, score


class ElasticsearchHandler:
    """
    Handles interaction with Elasticsearch for data storage and retrieval.
    
    This class manages the connection to Elasticsearch and provides methods for
    storing and querying network traffic data and detected threats.
    
    Attributes:
        es_client: Elasticsearch client instance
        index_name (str): Name of the Elasticsearch index for storing data
    """
    
    def __init__(self, hosts: List[str], index_name: str):
        """
        Initialize the Elasticsearch handler.
        
        Args:
            hosts: List of Elasticsearch host addresses
            index_name: Name of the index to use for storing data
        """
        self.es_client = Elasticsearch(hosts)
        self.index_name = index_name
        self._create_index()
        
    def _create_index(self) -> None:
        """Create the Elasticsearch index with appropriate mappings."""
        if not self.es_client.indices.exists(index=self.index_name):
            mappings = {
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "src_ip": {"type": "ip"},
                        "dst_ip": {"type": "ip"},
                        "protocol": {"type": "keyword"},
                        "src_port": {"type": "integer"},
                        "dst_port": {"type": "integer"},
                        "length": {"type": "integer"},
                        "is_threat": {"type": "boolean"},
                        "anomaly_score": {"type": "float"}
                    }
                }
            }
            self.es_client.indices.create(
                index=self.index_name,
                body=mappings
            )
            
    def store_packet_data(self, packet_data: Dict, threat_info: Tuple[bool, float]) -> None:
        """
        Store packet data and threat information in Elasticsearch.
        
        Args:
            packet_data: Dictionary containing packet information
            threat_info: Tuple of (is_threat, anomaly_score)
        """
        is_threat, anomaly_score = threat_info
        document = {
            **packet_data,
            'is_threat': is_threat,
            'anomaly_score': anomaly_score
        }
        self.es_client.index(
            index=self.index_name,
            body=document
        )
        
    def query_threats(self, time_range: Optional[Dict] = None) -> List[Dict]:
        """
        Query detected threats from Elasticsearch.
        
        Args:
            time_range: Optional dictionary with 'start' and 'end' timestamps
            
        Returns:
            List of detected threats matching the query
        """
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"is_threat": True}}
                    ]
                }
            }
        }
        
        if time_range:
            query["query"]["bool"]["must"].append({
                "range": {
                    "timestamp": {
                        "gte": time_range["start"],
                        "lte": time_range["end"]
                    }
                }
            })
            
        results = self.es_client.search(
            index=self.index_name,
            body=query
        )
        return [hit["_source"] for hit in results["hits"]["hits"]]


class AlertManager:
    """
    Manages the alerting system for detected threats.
    
    This class handles the generation and distribution of alerts when threats
    are detected in the network traffic.
    
    Attributes:
        alert_threshold (float): Threshold for generating alerts
        notification_channels (List): List of notification methods
    """
    
    def __init__(self, alert_threshold: float = -0.5):
        """
        Initialize the AlertManager.
        
        Args:
            alert_threshold: Threshold value for generating alerts
        """
        self.alert_threshold = alert_threshold
        self.notification_channels = []
        
    def add_notification_channel(self, channel: callable) -> None:
        """
        Add a notification channel for alerts.
        
        Args:
            channel: Callable that handles sending notifications
        """
        self.notification_channels.append(channel)
        
    def process_threat(self, packet_data: Dict, threat_info: Tuple[bool, float]) -> None:
        """
        Process a detected threat and generate alerts if necessary.
        
        Args:
            packet_data: Dictionary containing packet information
            threat_info: Tuple of (is_threat, anomaly_score)
        """
        is_threat, score = threat_info
        
        if is_threat and score < self.alert_threshold:
            alert_message = self._generate_alert_message(packet_data, score)
            self._send_alerts(alert_message)
            
    def _generate_alert_message(self, packet_data: Dict, score: float) -> str:
        """
        Generate an alert message for a detected threat.
        
        Args:
            packet_data: Dictionary containing packet information
            score: Anomaly score for the detected threat
            
        Returns:
            Formatted alert message string
        """
        return f"""
        SECURITY ALERT - Potential Threat Detected
        Timestamp: {packet_data['timestamp']}
        Source IP: {packet_data['src_ip']}
        Destination IP: {packet_data['dst_ip']}
        Protocol: {packet_data['protocol']}
        Anomaly Score: {score}
        """
        
    def _send_alerts(self, message: str) -> None:
        """
        Send alerts through all configured notification channels.
        
        Args:
            message: Alert message to send
        """
        for channel in self.notification_channels:
            try:
                channel(message)
            except Exception as e:
                logger.error(f"Failed to send alert through channel: {str(e)}")



class PacketAnalyzer:
    """
    Handles packet capture and initial analysis of network traffic.
    
    This class provides methods for capturing network packets, extracting relevant
    features, and preparing data for threat analysis.
    
    Attributes:
        interface (str): Network interface to monitor
        packet_buffer (List): Buffer to store captured packets for batch processing
        buffer_size (int): Maximum size of packet buffer before processing
    """
    
    def __init__(self, interface: str, threat_detector: ThreatDetector, 
                 es_handler: ElasticsearchHandler, alert_manager: AlertManager,
                 buffer_size: int = 1000):
        """
        Initialize the PacketAnalyzer.
        
        Args:
            interface: Network interface to monitor
            threat_detector: ThreatDetector instance for analyzing threats
            es_handler: ElasticsearchHandler instance for data storage
            alert_manager: AlertManager instance for handling alerts
            buffer_size: Size of packet buffer before processing
        """
        self.interface = interface
        self.packet_buffer = []
        self.buffer_size = buffer_size
        self.threat_detector = threat_detector
        self.es_handler = es_handler
        self.alert_manager = alert_manager
        
    def start_capture(self) -> None:
        """
        Start capturing packets on the specified interface.
        
        This method begins the packet capture process using Scapy's sniff function.
        Captured packets are processed through the _process_packet method.
        """
        logger.info(f"Starting packet capture on interface {self.interface}")
        sniff(iface=self.interface, prn=self._process_packet, store=False)
        
    def _process_packet(self, packet) -> Dict:
        """
        Process individual packets and extract relevant features.
        
        Args:
            packet: Captured network packet
            
        Returns:
            Dictionary containing extracted packet features
        """
        packet_data = {
            'timestamp': datetime.now().isoformat(),
            'protocol': None,
            'src_ip': None,
            'dst_ip': None,
            'src_port': None,
            'dst_port': None,
            'length': len(packet),
            'flags': None
        }
        
        if IP in packet:
            packet_data.update({
                'src_ip': packet[IP].src,
                'dst_ip': packet[IP].dst,
                'protocol': packet[IP].proto
            })
            
            if TCP in packet:
                packet_data.update({
                    'src_port': packet[TCP].sport,
                    'dst_port': packet[TCP].dport,
                    'flags': packet[TCP].flags
                })
            elif UDP in packet:
                packet_data.update({
                    'src_port': packet[UDP].sport,
                    'dst_port': packet[UDP].dport
                })
                
        self.packet_buffer.append(packet_data)
        
        if len(self.packet_buffer) >= self.buffer_size:
            self._process_buffer()
            
        return packet_data
    
    def _process_buffer(self) -> None:
        """
        Process the accumulated packet buffer.
        
        This method is called when the buffer reaches its maximum size.
        It performs threat detection, stores results in Elasticsearch,
        and generates alerts if necessary.
        """
        if self.packet_buffer:
            for packet_data in self.packet_buffer:
                # Perform threat detection
                threat_info = self.threat_detector.detect_threats(packet_data)
                
                # Store data in Elasticsearch
                self.es_handler.store_packet_data(packet_data, threat_info)
                
                # Process any detected threats
                self.alert_manager.process_threat(packet_data, threat_info)
                
            self.packet_buffer.clear()


def main():
    """
    Main entry point for the network traffic analyzer.
    
    This function initializes all components and starts the monitoring process.
    It also loads historical data for training the threat detection model.
    """

    threat_detector = ThreatDetector()
    es_handler = ElasticsearchHandler(
        hosts=["localhost:9200"],
        index_name="network_traffic"
    )
    alert_manager = AlertManager()
 
    alert_manager.add_notification_channel(lambda msg: print(f"ALERT: {msg}"))

    historical_data = es_handler.query_threats({
        "start": "now-30d",
        "end": "now"
    })
    if historical_data:
        threat_detector.train_model(historical_data)
        logger.info("Trained threat detection model with historical data")
    else:
        logger.warning("No historical data available for model training")

    packet_analyzer = PacketAnalyzer(
        interface="eth0",
        threat_detector=threat_detector,
        es_handler=es_handler,
        alert_manager=alert_manager
    )
    
    try:
        logger.info("Starting network traffic analyzer...")
        packet_analyzer.start_capture()
    except KeyboardInterrupt:
        logger.info("Stopping network traffic analyzer...")
        sys.exit(0)


if __name__ == "__main__":
    main()