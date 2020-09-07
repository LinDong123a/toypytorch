from collections import OrderedDict

from ..parameter import Parameter


class Module(object):
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

    def __getattr__(self, name: str):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]

        raise ValueError("{} object has no attribute {}".format(
                         self.__class__.__name__, name))

    def __setattr__(self, name: str, value):
        params = self.__dict__.get("_parameters")
        if isinstance(value, Parameter):
            if params is None:
                raise TypeError("super().__init__() must be called before "
                                "addding parameter")
            params[name] = value
        elif isinstance(value, self.__class__):
            modules = self.__dict__.get("_modules")
            if modules is None:
                raise TypeError("super().__init__() must be called before "
                                "addding modules")
                
            modules[name] = value
        else:
            object.__setattr__(self, name, value)
            
    def forward(self, *input, **kwargs):
        raise NotImplementedError

    def parameters(self, prefix: str = ''):
        for _, param in self.named_parameters(prefix=prefix):
            yield param

    def named_parameters(self, prefix: str = ''):
        yield from self._named_params(
            lambda module: module._parameters.items(),
            prefix
        )
    
    def _named_params(self, get_member_fn, prefix: str):
        memo = set()
        
        for module_prefix, module in self.named_modules(prefix=prefix):
            members = get_member_fn(module)
            for k, v in members:
                if v is None:
                    continue
                
                memo.add(v)
                yield module_prefix + ("." if module_prefix else "") + k, v
    
    def named_modules(self, memo: set = None, prefix: str = ''):
        if memo is None:
            memo = set()

        if self not in memo:
            memo.add(self)
            yield prefix, self 
            for name, module in self._modules.items():
                if module is None:
                    continue

                submodule_prefix = prefix + ("." if prefix else "") + name
                yield from module.named_modules(memo=memo, prefix=submodule_prefix)

