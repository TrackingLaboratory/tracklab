
def select_highest_voted_att(atts, atts_confidences=None):
        
    confidence_sum = {}
    atts_confidences = [1] * len(atts) if atts_confidences is None else atts_confidences
    
    # Iterate through the predictions to calculate the total confidence for each attribute
    for jn, conf in zip(atts, atts_confidences):
        if jn not in confidence_sum:
            confidence_sum[jn] = 0
        confidence_sum[jn] += conf
    
    # Find the attribute with the maximum total confidence
    if len(confidence_sum) == 0:
        return None
    max_confidence_att = max(confidence_sum, key=confidence_sum.get)
    return max_confidence_att