package edu.shu.abs.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import edu.shu.abs.entity.Collection;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

/**
 * <p>
 *  Mapper 接口
 * </p>
 *
 * @author zyh
 * @since 2025-05-26 11:05:54
 */
@Mapper
public interface CollectionMapper extends BaseMapper<Collection> {
    @Update("update collection set is_default_collection = false where owner_id = ${owner_id} and is_default_collection = true ")
    Integer resetDefaultCollection(@Param("owner_id") Long ownerId);
    
    @Select("Select collectionId,ownerId from collection where owner_id = ${owner_id} and is_default_collection = true ")
    Collection selectDefaultCollection(@Param("owner_id") Long ownerId);
    
}
