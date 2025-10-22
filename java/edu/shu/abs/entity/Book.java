package edu.shu.abs.entity;

import lombok.Data;
import lombok.experimental.Accessors;

@Data
@Accessors(chain = true)
public class Book {
	private String title;
	private String author;
	private String book_id;
	private String cover_image;
	private String publisher;
	private String score;
}
