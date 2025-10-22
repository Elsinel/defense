package edu.shu.abs.entity;

import lombok.Data;
import lombok.experimental.Accessors;

import java.util.List;

@Data
@Accessors(chain = true)
public class RecommendResponse {
	private String status;
	private String message;
	private List<Book> recommendations;
	
	// getter和setter
	public String getStatus() { return status; }
	public void setStatus(String status) { this.status = status; }
	
	public String getMessage() { return message; }
	public void setMessage(String message) { this.message = message; }
	
	public List<Book> getRecommendations() { return recommendations; }
	public void setRecommendations(List<Book> recommendations) { this.recommendations = recommendations; }
}
