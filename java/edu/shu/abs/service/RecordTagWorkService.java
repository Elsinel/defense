package edu.shu.abs.service;

import edu.shu.abs.common.base.BaseService;
import edu.shu.abs.entity.RecordTagWork;

/**
 * <p>
 *  服务类
 * </p>
 *
 * @author zyh
 */
public interface RecordTagWorkService extends BaseService<RecordTagWork> {

    boolean addNewTagRecord(long tagId, long workId);

    Boolean deleteRecordTagWorkListByWorkId(long workId);
}
