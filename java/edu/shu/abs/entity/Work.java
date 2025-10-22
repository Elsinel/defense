package edu.shu.abs.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import lombok.Data;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * <p>
 * 作品
 * </p>
 *
 * @author zyh
 * @since 2025-05-26 11:05:54
 */
@Data
@Accessors(chain = true)
public class Work implements Serializable {

    
    private static final long serialVersionUID = 1293712947787864128L;

    /**
     * 作品id
     */
    @TableId(value = "work_id", type = IdType.AUTO)
    private Long workId;

    /**
     * 标签
     */
    private String tags;

    /**
     * 作品名
     */
    private String workName;

    /**
     * 作者
     */
    private String author;

    /**
     * 作品简介
     */
    private String introduction;

    /**
     * 总评分
     */
    private int sumRating;

    /**
     * 总评分用户数
     */
    private int sumRatingUserNumber;

    /**
     * 出版社
     */
    private String publisher;

    /**
     * 封面地址(URL)
     */
    private String coverLink;

    /**
     * 是否被逻辑删除
     */
    private Boolean isDeleted;

    /**
     * 创建时间
     */
    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;

    /**
     * 更新时间
     */
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updateTime;
    
    
    private String text_weight;
    
    private String img_weight;
}
