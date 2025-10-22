package edu.shu.abs.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * <p>
 * 收藏夹与作品的收藏关系
 * </p>
 *
 * @author zyh
 * @since 2025-05-26 11:05:54
 */
@Data
@Accessors(chain = true)
@TableName("record_collection_work")
public class RecordCollectionWork implements Serializable {

    
    private static final long serialVersionUID = 1293712947787864128L;

    /**
     * 收藏记录id
     */
    @TableId(value = "record_id", type = IdType.AUTO)
    private Long recordId;

    /**
     * 收藏夹id
     */
    private Long collectionId;

    /**
     * 作品id
     */
    private Long workId;

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
}
