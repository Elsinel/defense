package edu.shu.abs.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import edu.shu.abs.entity.User;
import org.apache.ibatis.annotations.Mapper;

/**
 * <p>
 *  Mapper 接口
 * </p>
 *
 * @author zyh
 * @since 2025-05-26 11:05:54
 */
@Mapper
public interface UserMapper extends BaseMapper<User> {

}
