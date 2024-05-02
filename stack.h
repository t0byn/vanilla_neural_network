#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#define DEFAULT_ALIGNMENT 8

#define KB(x) ((x) * 1024)
#define MB(x) ((x) * KB(1024))

#define IS_POWER_OF_2(x) (((x) & ((x) - 1)) == 0)

int get_padding_with_header(uintptr_t address, int header_size, int align)
{
    assert(IS_POWER_OF_2(align));
    int padding = address & (align - 1);
    if (padding < header_size)
    {
        int remain = header_size - padding;
        if ((remain & (align - 1)) == 0)
        {
            padding += (remain / align) * align;
        }
        else
        {
            padding += ((remain / align) + 1) * align;
        }
    }
    return padding;
}

struct StackAllocHeader
{
    uint8_t padding;
    int prev_offset;
};

struct StackAllocator
{
    unsigned char* buffer;
    int size;
    int offset;
    int prev_offset;
};

void stack_init(StackAllocator* stack, void* buffer, int size)
{
    assert(stack != NULL);
    stack->buffer = (unsigned char*)buffer;
    stack->size = size;
    stack->offset = 0;
    stack->prev_offset = 0;
}

void* stack_alloc(StackAllocator* stack, int size, int align = DEFAULT_ALIGNMENT)
{
    assert(size > 0);

    uint8_t max_align = 1 << (sizeof(StackAllocHeader::padding) * 8 - 1);
    if (align > max_align)
    {
        align = max_align;
    }

    uintptr_t start = (uintptr_t)stack->buffer + stack->offset;
    int padding = get_padding_with_header(start, sizeof(StackAllocHeader), align);
    int require_size = padding + size;

    if (start + require_size > (uintptr_t)stack->buffer + stack->size)
    {
        assert(0);
        fprintf(stderr, 
            "stack_alloc failed! request size: %d, padding size: %d, stack remaining size: %d", 
            size, padding, stack->size - stack->offset);
        return NULL;
    }

    unsigned char* ptr = &stack->buffer[stack->offset + padding];

    StackAllocHeader* header = (StackAllocHeader*)(ptr - sizeof(StackAllocHeader));
    header->padding = padding;
    header->prev_offset = stack->prev_offset;

    stack->prev_offset = stack->offset;
    stack->offset += require_size;

    return memset((void*)ptr, 0, size);
}

void stack_free(StackAllocator* stack, void* ptr)
{
    if (ptr < stack->buffer || ptr >= stack->buffer + stack->size)
    {
        assert(0);
        fprintf(stderr, 
            "stack_free error! ptr not in stack scope. ptr: %ulld, stack range: [%ulld, %ulld)", 
            (uintptr_t)ptr, (uintptr_t)stack->buffer, (uintptr_t)stack->buffer + stack->size);
        return;
    }

    // double free
    if (ptr >= stack->buffer + stack->offset)
    {
        assert(0);
        return;
    }

    // prevent out of order free
    StackAllocHeader* header = (StackAllocHeader*)((unsigned char*)ptr - sizeof(StackAllocHeader));
    int prev_offset = (uintptr_t)ptr - (uintptr_t)stack->buffer - header->padding;
    if (prev_offset != stack->prev_offset)
    {
        assert(0);
        fprintf(stderr, "stack_free error! out of order free!");
        return;
    }

    stack->offset = stack->prev_offset;
    stack->prev_offset = header->prev_offset;
}

void stack_free_all(StackAllocator* stack)
{
    assert(stack != NULL);
    stack->offset = 0;
}