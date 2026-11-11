/* File: tests/c/smoke.c
 * Purpose: a C caller of the shared library, built and run by tests/c_smoke.rs.
 *
 * Main responsibilities: encode and pack through the public header, read back by offset and by tensor
 * name into malloc'd memory, describe and verify members, restore, edit and recover a pack, and check
 * an error path.
 *
 * Usage: smoke <input> <tensor name> <scratch directory>. Exits 0 when every check passed, and writes
 * only under the scratch directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ziptensors.h"

/* Ends the run, naming the check and the library's last error. */
static void die(const char *what) {
    char text[512];
    (void)zt_last_error(text, sizeof text);
    fprintf(stderr, "smoke: %s: %s\n", what, text);
    exit(1);
}

#define CHECK_OK(call)                                                                             \
    do {                                                                                           \
        if ((call) != ZT_STATUS_OK) die(#call);                                                    \
    } while (0)

#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) die(#condition);                                                         \
    } while (0)

/* Reads a whole file into malloc'd memory. */
static unsigned char *read_file(const char *path, size_t *len) {
    FILE *file = fopen(path, "rb");
    CHECK(file != NULL);
    CHECK(fseek(file, 0, SEEK_END) == 0);
    long size = ftell(file);
    CHECK(size >= 0);
    CHECK(fseek(file, 0, SEEK_SET) == 0);
    unsigned char *bytes = malloc((size_t)size + 1);
    CHECK(bytes != NULL);
    CHECK(fread(bytes, 1, (size_t)size, file) == (size_t)size);
    fclose(file);
    *len = (size_t)size;
    return bytes;
}

/* Joins a scratch directory and a file name. */
static char *join(const char *dir, const char *name) {
    size_t len = strlen(dir) + strlen(name) + 2;
    char *path = malloc(len);
    CHECK(path != NULL);
    snprintf(path, len, "%s/%s", dir, name);
    return path;
}

/* Checks that the member holds exactly the source bytes, reading into uninitialized memory. */
static void check_bytes(const zt_member *member, const unsigned char *source, size_t source_len) {
    uint64_t len = 0;
    CHECK_OK(zt_member_len(member, &len));
    CHECK(len == source_len);
    unsigned char *dst = malloc(source_len);
    CHECK(dst != NULL);
    CHECK_OK(zt_read_at(member, 0, dst, source_len));
    CHECK(memcmp(dst, source, source_len) == 0);
    free(dst);
}

/* Finds the tensor by name in the table, then decodes it and compares it with the source. */
static void check_tensor(const zt_member *member, const char *name, const unsigned char *source) {
    size_t count = 0;
    CHECK_OK(zt_tensor_count(member, &count));
    for (size_t index = 0; index < count; index++) {
        zt_tensor_desc desc;
        CHECK_OK(zt_tensor_info(member, index, &desc));
        if (strcmp(desc.name, name) != 0) continue;
        CHECK(desc.name_len == strlen(name));
        CHECK(desc.ndim > 0 && desc.shape[0] > 0);
        size_t len = (size_t)desc.len;
        unsigned char *dst = malloc(len);
        CHECK(dst != NULL);
        CHECK_OK(zt_decode_tensor_into(member, name, dst, len));
        CHECK(memcmp(dst, source + desc.offset, len) == 0);
        free(dst);
        return;
    }
    die("the tensor is not in the table");
}

int main(int argc, char **argv) {
    CHECK(argc == 4);
    const char *input = argv[1];
    const char *tensor = argv[2];
    const char *scratch = argv[3];
    size_t source_len = 0;
    unsigned char *source = read_file(input, &source_len);

    CHECK(zt_abi_version() == ZT_ABI_VERSION);
    CHECK_OK(zt_enable_stderr_logging(ZT_LOG_LEVEL_ERROR));
    CHECK_OK(zt_enable_stderr_logging(ZT_LOG_LEVEL_ERROR));

    /* One member, through the one-shot encoder. Fields left out take their defaults. */
    char *single = join(scratch, "single.ziptensors");
    zt_encode_options encode = {.struct_size = sizeof encode, .threads = 1, .mode = ZT_MODE_FAST};
    CHECK_OK(zt_encode_file(input, single, &encode));

    zt_open_options open = {.struct_size = sizeof open, .threads = 1, .verify = ZT_VERIFY_CHUNK};
    zt_container *container = NULL;
    CHECK_OK(zt_open(single, &open, &container));
    zt_member *member = NULL;
    CHECK_OK(zt_member_open(container, NULL, &member));
    check_bytes(member, source, source_len);
    check_tensor(member, tensor, source);

    char *restored_path = join(scratch, "restored.bin");
    CHECK_OK(zt_restore_file(member, restored_path));
    size_t restored_len = 0;
    unsigned char *restored = read_file(restored_path, &restored_len);
    CHECK(restored_len == source_len && memcmp(restored, source, source_len) == 0);

    /* The member list, and a dtype constant from the header. */
    size_t member_count = 0;
    CHECK_OK(zt_member_count(container, &member_count));
    CHECK(member_count == 1);
    const char *member_name = NULL;
    size_t member_name_len = 0;
    CHECK_OK(zt_member_name(container, 0, &member_name, &member_name_len));
    CHECK(member_name_len == strlen(member_name));
    CHECK(zt_member_name(container, 1, &member_name, &member_name_len) ==
          ZT_STATUS_INVALID_ARGUMENT);
    CHECK(ZT_DTYPE_F32 == 0x0012 && ZT_DTYPE_Q8_0 == 0x0030);

    /* What the index states about the member, and a full decode against its hash. */
    zt_member_desc info;
    memset(&info, 0xA5, sizeof info);
    info.struct_size = sizeof info;
    CHECK_OK(zt_member_info(container, 0, &info));
    CHECK(info.len == source_len);
    CHECK(info.profile == ZT_PROFILE_SAFETENSORS);
    CHECK(info.tensor_count == 1);
    info.struct_size = sizeof info - 1;
    CHECK(zt_member_info(container, 0, &info) == ZT_STATUS_STRUCT_SIZE);
    CHECK_OK(zt_member_verify(member));

    /* The member keeps the data alive past the container. */
    zt_close(container);
    check_bytes(member, source, source_len);
    zt_member_close(member);

    /* Two members, through the builder, with default options. */
    char *packed = join(scratch, "pack.ziptensors");
    zt_pack *pack = NULL;
    CHECK_OK(zt_pack_new(packed, NULL, &pack));
    CHECK_OK(zt_pack_add_file(pack, input, "models/first.bin"));
    CHECK_OK(zt_pack_add_file(pack, input, "models/second.bin"));
    CHECK_OK(zt_pack_finish(pack));
    CHECK(zt_pack_finish(pack) == ZT_STATUS_INVALID_STATE);
    zt_pack_free(pack);

    CHECK_OK(zt_open(packed, NULL, &container));
    CHECK(zt_member_open(container, NULL, &member) == ZT_STATUS_MEMBER_NOT_FOUND);
    CHECK(member == NULL);
    CHECK_OK(zt_member_open(container, "models/second.bin", &member));
    check_bytes(member, source, source_len);
    zt_member_close(member);

    /* Every member restored under a new directory. */
    char *tree = join(scratch, "tree");
    uint64_t restored_bytes = 0;
    CHECK_OK(zt_restore_all(container, tree, &restored_bytes));
    CHECK(restored_bytes == 2 * (uint64_t)source_len);
    zt_close(container);

    /* An append that removes one member and adds another, then a refused kind and a check. */
    zt_edit_op ops[2] = {
        {.kind = ZT_EDIT_REMOVE, .name = "models/first.bin"},
        {.kind = ZT_EDIT_ADD, .name = "models/third.bin", .path = input},
    };
    zt_edit_options edit = {
        .struct_size = sizeof edit, .strategy = ZT_STRATEGY_APPEND, .mode = ZT_MODE_FAST, .threads = 1};
    CHECK_OK(zt_edit(packed, ops, 2, &edit));
    zt_edit_op bad = {.kind = 99, .name = "models/third.bin"};
    CHECK(zt_edit(packed, &bad, 1, &edit) == ZT_STATUS_INVALID_ARGUMENT);
    edit.repack = 1;
    CHECK(zt_edit(packed, ops + 1, 1, &edit) == ZT_STATUS_EDIT_REFUSED);

    CHECK_OK(zt_open(packed, NULL, &container));
    CHECK_OK(zt_member_count(container, &member_count));
    CHECK(member_count == 2);
    CHECK_OK(zt_member_name(container, 1, &member_name, &member_name_len));
    CHECK(strcmp(member_name, "models/third.bin") == 0);
    CHECK_OK(zt_member_open(container, "models/third.bin", &member));
    check_bytes(member, source, source_len);
    zt_member_close(member);
    zt_close(container);

    /* The edited file opens as it is, so recovery keeps its whole length. */
    zt_recover_options recover = {.struct_size = sizeof recover, .dry_run = 1};
    uint64_t valid_len = 0;
    CHECK_OK(zt_recover(packed, &recover, &valid_len));
    CHECK(valid_len > 0);

    /* A failure, with the text sized by asking and then cut. */
    char *missing = join(scratch, "missing.ziptensors");
    CHECK(zt_open(missing, NULL, &container) == ZT_STATUS_IO);
    CHECK(container == NULL);
    size_t needed = zt_last_error(NULL, 0);
    CHECK(needed > 8);
    char *text = malloc(needed + 1);
    CHECK(text != NULL);
    CHECK(zt_last_error(text, needed + 1) == needed);
    CHECK(strlen(text) == needed);
    CHECK(strstr(text, "missing.ziptensors") != NULL);
    char cut[8];
    CHECK(zt_last_error(cut, sizeof cut) == needed);
    CHECK(strlen(cut) == sizeof cut - 1);

    zt_close(NULL);
    zt_member_close(NULL);
    zt_pack_free(NULL);

    free(text);
    free(missing);
    free(tree);
    free(packed);
    free(restored);
    free(restored_path);
    free(single);
    free(source);
    return 0;
}
