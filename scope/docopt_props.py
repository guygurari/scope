from docopt import docopt


class DocoptProperties:
    '''Use docopt to parse command-line argument, and store them in
    properties.'''
    def __init__(self, doc_string=__doc__,
                 str_to_obj=True, add_flipped_nos=True):
        '''Parse args from the given doc string.
        If str_to_obj is True, try to convert strings to
        integers/floats/booleans if possible.
        If add_flipped_nos is True, arguments of the form --no-something
        will also receive properties of the form --something with negated
        values.'''
        args = docopt(doc_string)
        self.flipped_nos = add_flipped_nos
        for key, value in args.items():
            prop_name = self._to_prop_name(key)
            prop_value = self._to_prop_value(value, str_to_obj)
            self._add_property(prop_name, prop_value)
            if add_flipped_nos and self._is_no(prop_name):
                assert type(prop_value) == bool
                self._add_property(prop_name[3:], not prop_value)

    def set_property(self, name, value):
        '''Change the value of an existing property.'''
        self._add_property(name, value)

    def _add_property(self, name, value):
        exec('self.{} = {}'.format(name, value))

    def _is_no(self, name):
        return name.startswith('no_')

    def _to_prop_name(self, key):
        return key[2:].replace('-', '_')

    def _to_prop_value(self, value, str_to_obj):
        if type(value) == bool:
            return value
        elif type(value) == str:
            if str_to_obj:
                if value.lower() == 'false':
                    return False
                elif value.lower() == 'true':
                    return True
                elif '.' in value or 'e' in value:
                    # Parse as float
                    try:
                        return float(value)
                    except ValueError:
                        pass
                else:
                    # Parse as int
                    try:
                        return int(value)
                    except ValueError:
                        pass
            # Get here if parsing didn't work
            return "'{}'".format(value)
        else:
            raise ValueError('Unexpected value {} of type {}'.format(
                value, type(value)))

    def __str__(self):
        s = ''
        for property, value in vars(self).items():
            if self.flipped_nos and self._is_no(property):
                continue
            s += '{}:  {}\n'.format(property, value)
        return s
