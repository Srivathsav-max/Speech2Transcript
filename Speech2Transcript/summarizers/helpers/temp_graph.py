class TemporalGraph:
    """Represents medical events in a temporal graph for enhanced reasoning"""
    
    def __init__(self):
        self.nodes = {}  # Event nodes
        self.edges = []  # Temporal relationships
    
    def build_from_entities(self, entities, text):
        """Build a temporal graph from medical entities"""
        # Create nodes for each entity
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                # Skip uncertain or hypothetical entities
                if entity.is_uncertain or entity.is_hypothetical:
                    continue
                
                # Create a unique ID for this entity
                if entity.position:
                    entity_id = f"{entity_type}_{entity.position[0]}_{entity.position[1]}"
                else:
                    # Use a hash if no position is available
                    entity_id = f"{entity_type}_{hash(entity.value) % 10000}"
                
                # Add as node
                self.nodes[entity_id] = {
                    "type": entity_type,
                    "value": entity.value,
                    "normalized_value": entity.normalized_value,
                    "temporal_context": entity.temporal_context,
                    "position": entity.position,
                    "metadata": entity.metadata
                }
        
        # Define temporal relationships
        self._add_relationships()
        
        # Resolve potential temporal conflicts
        self._resolve_conflicts()
        
        return self
    
    def _add_relationships(self):
        """Add temporal relationships between nodes"""
        # Define temporal contexts in chronological order
        temporal_order = [
            "past_history", "recent_past", "current", 
            "immediate_future", "distant_future"
        ]
        
        # For each pair of nodes, define temporal relationships
        node_ids = list(self.nodes.keys())
        
        for i, node1_id in enumerate(node_ids):
            for node2_id in node_ids[i+1:]:
                node1 = self.nodes[node1_id]
                node2 = self.nodes[node2_id]
                
                # Skip if either node doesn't have temporal context
                if not node1.get("temporal_context") or not node2.get("temporal_context"):
                    continue
                
                # Get indices in temporal order
                try:
                    idx1 = temporal_order.index(node1["temporal_context"])
                    idx2 = temporal_order.index(node2["temporal_context"])
                    
                    if idx1 < idx2:
                        relation = "before"
                    elif idx1 > idx2:
                        relation = "after"
                    else:
                        relation = "concurrent"
                    
                    self.edges.append({
                        "source": node1_id,
                        "target": node2_id,
                        "relation": relation
                    })
                    
                except ValueError:
                    # Skip if temporal context not in our defined order
                    continue
    
    def _resolve_conflicts(self):
        """Resolve potential conflicts in temporal relationships"""
        # Simple check for direct contradictions
        contradictions = []
        
        for edge1 in self.edges:
            for edge2 in self.edges:
                if edge1["source"] == edge2["target"] and edge1["target"] == edge2["source"]:
                    if edge1["relation"] == "before" and edge2["relation"] == "before":
                        contradictions.append((edge1, edge2))
        
        # Handle contradictions
        for edge1, edge2 in contradictions:
            if edge1 in self.edges:
                self.edges.remove(edge1)
            if edge2 in self.edges:
                self.edges.remove(edge2)
    
    def get_timeline(self):
        """Generate a timeline of medical events"""
        # Group nodes by temporal context
        timeline = {}
        
        for node_id, node in self.nodes.items():
            temporal_context = node.get("temporal_context", "unknown")
            
            if temporal_context not in timeline:
                timeline[temporal_context] = []
            
            timeline[temporal_context].append({
                "id": node_id,
                "type": node["type"],
                "value": node["value"],
                "normalized_value": node["normalized_value"]
            })
        
        return timeline