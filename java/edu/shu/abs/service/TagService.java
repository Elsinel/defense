package edu.shu.abs.service;

import edu.shu.abs.common.base.BaseService;
import edu.shu.abs.entity.Tag;

/**
 * <p>
 *  服务类
 * </p>
 *
 * @author zyh
 */
public interface TagService extends BaseService<Tag> {

    Long saveIfNotExist(String tagName);
}
